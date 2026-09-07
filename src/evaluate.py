"""Metriques d'evaluation adaptees au desequilibre : PR-AUC, calibration, lift, importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def lift_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    r = pd.DataFrame({'y': np.asarray(y_true), 'p': np.asarray(proba)})
    r['bin'] = pd.qcut(r['p'].rank(method='first'), n_bins, labels=False)
    t = r.groupby('bin')['y'].agg(['count', 'sum', 'mean']).sort_index(ascending=False)
    t['capture_%'] = (100 * t['sum'] / max(r['y'].sum(), 1)).round(1)
    return t


def metrics(y_true, proba) -> dict:
    lt = lift_table(y_true, proba)
    return {
        'pr_auc': round(float(average_precision_score(y_true, proba)), 4),
        'roc_auc': round(float(roc_auc_score(y_true, proba)), 4),
        'brier': round(float(brier_score_loss(y_true, proba)), 5),
        'base_rate': round(float(np.mean(y_true)), 4),
        'top_decile_capture_%': float(lt['capture_%'].iloc[0]),
        'top30_capture_%': round(float(lt['capture_%'].iloc[:3].sum()), 1),
    }


def gain_importance(raw_model, features: list[str], n: int = 12) -> dict:
    imp = pd.DataFrame({'feature': features, 'gain': raw_model.booster_.feature_importance('gain')})
    imp = imp.sort_values('gain', ascending=False).head(n)
    return {row.feature: round(float(row.gain), 1) for row in imp.itertuples()}
