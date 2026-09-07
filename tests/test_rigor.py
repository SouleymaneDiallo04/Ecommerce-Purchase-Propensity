import numpy as np

from src.business import decile_targeting, expected_value_threshold
from src.data import make_synthetic
from src.features import build_customer_features, feature_columns
from src.validation import temporal_cv


def test_temporal_cv_runs():
    s, _ = make_synthetic(n_customers=3000, seed=5)
    cust = build_customer_features(s)
    cat, num = feature_columns(cust)
    cv = temporal_cv(cust, cat, num, n_splits=3, model_params={'n_estimators': 120})
    assert cv['n_folds'] >= 1
    assert 0.0 <= cv['roc_auc_mean'] <= 1.0


def test_expected_value_threshold():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    p = np.array([.1, .2, .9, .3, .8, .7, .05, .4, .6, .15])
    r = expected_value_threshold(y, p, value_tp=100, cost_action=5)
    assert 0.0 <= r['threshold'] <= 1.0
    assert r['buyers_captured'] <= int(y.sum())


def test_decile_targeting():
    y = np.array([0, 1] * 50)
    p = np.linspace(0, 1, 100)
    r = decile_targeting(y, p, top_fraction=0.3)
    assert 0.0 <= r['buyers_capture_%'] <= 100.0
