import pandas as pd
import numpy as np

from src.business import decile_targeting, expected_value_threshold
from src.data import make_synthetic
from src.features import build_customer_features, build_insession_features, feature_columns
from src.validation import temporal_cv

# Totaux de session mecaniquement gonfles par l'achat : ne doivent JAMAIS servir de feature.
FORBIDDEN = {'pageviews', 'hits', 'time_on_site', 'bounces'}


def test_no_engagement_leak_in_features():
    """Aucune variable d'engagement (fuite) ni page post-intention dans les features."""
    s, h = make_synthetic(n_customers=800, seed=11)
    cat_c, num_c = feature_columns(build_customer_features(s))
    assert FORBIDDEN.isdisjoint(set(cat_c) | set(num_c)), 'fuite: totaux de session en features (customer)'
    cat_i, num_i = feature_columns(build_insession_features(s, h))
    assert FORBIDDEN.isdisjoint(set(cat_i) | set(num_i)), 'fuite: totaux de session en features (in-session)'
    # les pages posterieures a l'intention (panier/checkout/achat) ne doivent pas apparaitre
    assert not ({'pre_cart', 'pre_checkout', 'pre_purchase'} & (set(cat_i) | set(num_i))), \
        'fuite: page post-intention dans la fenetre de debut de parcours'


def _mini_sessions():
    """Un visiteur, 3 sessions datees, converted = [1, 0, 1] (colonnes minimales requises)."""
    const = dict(channelGrouping='Direct', source='Direct', medium='(none)', device_category='desktop',
                 os='Windows', country='France', sub_continent='Western Europe',
                 is_mobile=0, new_visit=0, hour=12)
    rows = [
        dict(session_id='s1', fullVisitorId='v1', date='20170101', visit_number=1, pageviews=3, converted=1, **const),
        dict(session_id='s2', fullVisitorId='v1', date='20170115', visit_number=2, pageviews=2, converted=0, **const),
        dict(session_id='s3', fullVisitorId='v1', date='20170201', visit_number=3, pageviews=4, converted=1, **const),
    ]
    return pd.DataFrame(rows)


def test_customer_features_are_strictly_prior():
    """L'historique n'utilise QUE le passe : ni la session courante, ni le futur ne fuient."""
    cust = build_customer_features(_mini_sessions()).set_index('session_id')
    # 1re session du visiteur : aucun passe
    assert cust.loc['s1', 'prior_sessions'] == 0
    assert cust.loc['s1', 'prior_purchases'] == 0
    assert cust.loc['s1', 'is_returning'] == 0
    assert cust.loc['s1', 'days_since_last'] == -1
    # s2 ne voit que s1 (qui a converti)
    assert cust.loc['s2', 'prior_sessions'] == 1
    assert cust.loc['s2', 'prior_purchases'] == 1
    # s3 a converti, mais son propre label ne compte pas : il ne voit que s1 (converti) et s2 (non)
    assert cust.loc['s3', 'prior_sessions'] == 2
    assert cust.loc['s3', 'prior_purchases'] == 1
    # invariant global : jamais plus d'achats passes que de sessions passees
    assert (cust['prior_purchases'] <= cust['prior_sessions']).all()


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
