"""Traduction du score en valeur business : seuil optimal cout/gain et ciblage par decile.

Un modele de propension ne sert pas a maximiser un F1 : il sert a decider QUI cibler pour
maximiser un gain (revenu attendu par acheteur capte) net du cout d'action (contact, remise).
"""
from __future__ import annotations

import numpy as np


def expected_value_threshold(y_true, proba, value_tp: float = 100.0, cost_action: float = 5.0) -> dict:
    """Seuil qui maximise la valeur attendue : value_tp * acheteurs_captes - cost_action * cibles.

    value_tp    : gain moyen d'un acheteur correctement cible.
    cost_action : cout d'une action marketing sur une session ciblee.
    """
    y = np.asarray(y_true)
    p = np.asarray(proba)
    order = np.argsort(-p)
    y_sorted, p_sorted = y[order], p[order]
    tp = np.cumsum(y_sorted)
    n_targeted = np.arange(1, len(y_sorted) + 1)
    ev = value_tp * tp - cost_action * n_targeted
    k = int(np.argmax(ev))
    return {
        'threshold': round(float(p_sorted[k]), 4),
        'expected_value': round(float(ev[k]), 1),
        'n_targeted': int(k + 1),
        'share_targeted_%': round(100.0 * (k + 1) / len(y_sorted), 1),
        'buyers_captured': int(tp[k]),
        'buyers_capture_%': round(100.0 * tp[k] / max(y.sum(), 1), 1),
        'params': {'value_tp': value_tp, 'cost_action': cost_action},
    }


def decile_targeting(y_true, proba, top_fraction: float = 0.3) -> dict:
    """Combien d'acheteurs capte-t-on en ciblant les `top_fraction` sessions les mieux scorees."""
    y = np.asarray(y_true)
    p = np.asarray(proba)
    order = np.argsort(-p)
    cut = int(top_fraction * len(y))
    captured = int(y[order][:cut].sum())
    lift = (captured / max(cut, 1)) / max(y.mean(), 1e-9)
    return {
        'top_fraction_%': round(100 * top_fraction, 0),
        'buyers_capture_%': round(100.0 * captured / max(y.sum(), 1), 1),
        'lift_vs_random': round(float(lift), 2),
    }
