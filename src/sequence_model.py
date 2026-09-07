"""Modele de sequence (GRU) sur le parcours de navigation.

Lit la suite ORDONNEE des pages d'une session (via page_token, ex. GA pagePathLevel1), hors
pages post-panier (anti-fuite), et predit la conversion. Sorties recalibrees (isotonic).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from .evaluate import metrics as eval_metrics

# Detection anti-fuite : actions e-commerce (panier/checkout/achat) + mots-cles de pages
# post-intention. On TRONQUE la sequence au premier hit d'intention (on ne garde que le
# prefixe de navigation pur), ce qui elimine panier, compte, confirmation, etc.
POST_INTENT_ACTIONS = {'3', '5', '6'}
POST_INTENT_KEYWORDS = ('basket', 'cart', 'checkout', 'payment', 'yourinfo', 'revieworder',
                        'ordercompleted', 'myaccount', 'registersuccess', 'signin', 'register',
                        'purchase')


def _post_intent_mask(hits: pd.DataFrame) -> pd.Series:
    kw = '|'.join(POST_INTENT_KEYWORDS)
    by_action = hits['action_type'].astype(str).isin(POST_INTENT_ACTIONS)
    by_token = hits['page_token'].astype(str).str.lower().str.contains(kw, regex=True, na=False)
    return by_action | by_token


def build_sequences(sessions: pd.DataFrame, hits: pd.DataFrame, max_len: int = 20, top_k: int = 40):
    """Renvoie X (n, max_len) d'ids de pages, y, dates, vocab. Prefixe de navigation avant intention.

    page_token = libelle riche de page. Vocabulaire plafonne aux `top_k` pages les plus frequentes.
    """
    h = hits.sort_values(['session_id', 'hit_index']).copy()
    h['_post'] = _post_intent_mask(h)
    first_post = h[h['_post']].groupby('session_id')['hit_index'].min()
    h = h.merge(first_post.rename('_first'), on='session_id', how='left')
    pre = h[h['_first'].isna() | (h['hit_index'] < h['_first'])]
    seqs = pre.groupby('session_id')['page_token'].apply(list)

    counts = Counter(tok for s in seqs for tok in s)
    top = [w for w, _ in counts.most_common(top_k)]
    vocab = {'<pad>': 0, '<unk>': 1}
    for w in top:
        vocab.setdefault(w, len(vocab))

    sess = sessions[['session_id', 'date', 'converted']].copy()
    mapped = sess['session_id'].map(seqs)
    sess['seq'] = [s if isinstance(s, list) else [] for s in mapped]

    def encode(s):
        ids = [vocab.get(p, 1) for p in s][:max_len]
        return ids + [0] * (max_len - len(ids))

    X = np.array([encode(s) for s in sess['seq']], dtype=np.int64)
    y = sess['converted'].to_numpy(dtype=np.float32)
    dates = sess['date'].astype(str).to_numpy()
    return X, y, dates, vocab


class SeqGRU(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 24, hid: int = 48):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.fc(h[-1]).squeeze(1)


def _predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def train_sequence_model(sessions, hits, split_date: str, out_dir: Path,
                         max_len: int = 20, epochs: int = 8, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, y, dates, vocab = build_sequences(sessions, hits, max_len)
    tr, te = dates < split_date, dates >= split_date
    Xtr_all, ytr_all = X[tr], y[tr]

    # split fit / calibration
    idx = np.arange(len(Xtr_all))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    fit_i, cal_i = idx[:cut], idx[cut:]

    model = SeqGRU(len(vocab))
    pos = float(ytr_all[fit_i].sum())
    neg = len(fit_i) - pos
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0)))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr_all[fit_i]), torch.tensor(ytr_all[fit_i]))
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()

    # calibration isotonic sur le pli de calibration
    p_cal = _predict(model, Xtr_all[cal_i])
    iso = IsotonicRegression(out_of_bounds='clip').fit(p_cal, ytr_all[cal_i])

    p_raw = _predict(model, X[te])
    p = iso.transform(p_raw)
    mets = eval_metrics(y[te], p)
    mets['brier_raw'] = round(float(brier_score_loss(y[te], p_raw)), 5)
    mets.update(n_train=int(tr.sum()), n_test=int(te.sum()), epochs=epochs,
                max_len=max_len, vocab_size=len(vocab))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(out_dir) / 'sequence.pt')
    joblib.dump({'vocab': vocab, 'max_len': max_len, 'calibrator': iso}, Path(out_dir) / 'sequence_meta.joblib')
    return {'metrics': mets}
