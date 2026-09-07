"""Attribution Markov (removal effect) : impact reel de chaque canal dans les parcours.

Methode : on modelise les parcours (start -> canaux -> conversion / null) par une chaine de
Markov absorbante. L'importance d'un canal = chute de la probabilite de conversion quand on
le retire du graphe (removal effect).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_paths(sessions: pd.DataFrame) -> pd.DataFrame:
    df = sessions.sort_values(['fullVisitorId', 'date'])
    g = df.groupby('fullVisitorId')
    return pd.DataFrame({'path': g['channelGrouping'].apply(list), 'conv': g['converted'].max()})


def _matrix(paths: pd.DataFrame):
    channels = sorted({c for p in paths['path'] for c in p})
    states = ['start'] + channels + ['conv', 'null']
    idx = {s: i for i, s in enumerate(states)}
    T = np.zeros((len(states), len(states)))
    for path, conv in zip(paths['path'], paths['conv']):
        seq = ['start'] + list(path) + (['conv'] if conv else ['null'])
        for a, b in zip(seq[:-1], seq[1:]):
            T[idx[a], idx[b]] += 1
    T[idx['conv'], idx['conv']] = 1.0
    T[idx['null'], idx['null']] = 1.0
    return T, states, idx, channels


def _conv_prob(P: np.ndarray, states: list[str], idx: dict) -> float:
    absorbing = [idx['conv'], idx['null']]
    transient = [i for i in range(len(states)) if i not in absorbing]
    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]
    try:
        N = np.linalg.inv(np.eye(len(transient)) - Q)
    except np.linalg.LinAlgError:
        return float('nan')
    B = N @ R
    return float(B[transient.index(idx['start']), absorbing.index(idx['conv'])])


def _normalize(T: np.ndarray) -> np.ndarray:
    row = T.sum(1, keepdims=True)
    row[row == 0] = 1.0
    return T / row


def markov_attribution(sessions: pd.DataFrame) -> dict:
    paths = build_paths(sessions)
    T, states, idx, channels = _matrix(paths)
    base = _conv_prob(_normalize(T), states, idx)
    removal = {}
    for ch in channels:
        T2 = T.copy()
        j = idx[ch]
        into = T2[:, j].copy()          # transitions entrant dans le canal -> redirigees vers null
        T2[:, j] = 0.0
        T2[:, idx['null']] += into
        T2[j, :] = 0.0                  # une fois dans le canal (retire) -> null
        T2[j, idx['null']] = 1.0
        conv2 = _conv_prob(_normalize(T2), states, idx)
        removal[ch] = max(base - conv2, 0.0) if conv2 == conv2 else 0.0  # nan-safe
    total = sum(removal.values()) or 1.0
    n_conv = int(paths['conv'].sum())
    channels_out = {ch: {'removal_effect': round(v, 4),
                         'attributed_conversions': round(v / total * n_conv, 1)}
                    for ch, v in sorted(removal.items(), key=lambda kv: -kv[1])}
    return {'base_conversion_prob': round(float(base), 4), 'n_conversions': n_conv, 'channels': channels_out}
