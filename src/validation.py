"""Validation croisee temporelle (fenetre glissante extensible).

Plus honnete qu'un seul split : on entraine sur le passe, on teste sur la periode suivante,
plusieurs fois, et on rapporte moyenne et ecart-type. Evite de sur-interpreter un split chanceux.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import metrics as eval_metrics
from .model import make_model, prepare_categories


def temporal_folds(dates: np.ndarray, n_splits: int = 4):
    """Decoupe les dates uniques en n_splits+1 blocs ; le fold i entraine sur les blocs 0..i,
    teste sur le bloc i+1 (fenetre extensible)."""
    uniq = np.sort(np.unique(dates))
    blocks = np.array_split(uniq, n_splits + 1)
    folds = []
    for i in range(n_splits):
        train_dates = set(np.concatenate(blocks[:i + 1]).tolist())
        test_dates = set(blocks[i + 1].tolist())
        folds.append((train_dates, test_dates))
    return folds


def temporal_cv(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str],
                n_splits: int = 4, model_params: dict | None = None) -> dict:
    feats = cat_cols + num_cols
    mp = model_params or {'n_estimators': 300}
    scores = []
    for tr_dates, te_dates in temporal_folds(df['date'].to_numpy(), n_splits):
        tr = df[df['date'].isin(tr_dates)].copy()
        te = df[df['date'].isin(te_dates)].copy()
        if tr['converted'].sum() < 5 or te['converted'].sum() < 5:
            continue
        tr, te = prepare_categories(tr, te, cat_cols)
        m = make_model(**mp)
        m.fit(tr[feats], tr['converted'], categorical_feature=cat_cols)
        p = m.predict_proba(te[feats])[:, 1]
        scores.append(eval_metrics(te['converted'], p))
    if not scores:
        return {'n_folds': 0}
    pr = [s['pr_auc'] for s in scores]
    roc = [s['roc_auc'] for s in scores]
    return {
        'n_folds': len(scores),
        'pr_auc_mean': round(float(np.mean(pr)), 4), 'pr_auc_std': round(float(np.std(pr)), 4),
        'roc_auc_mean': round(float(np.mean(roc)), 4), 'roc_auc_std': round(float(np.std(roc)), 4),
    }
