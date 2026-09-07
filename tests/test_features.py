from src.data import make_synthetic
from src.features import build_customer_features, build_insession_features


def test_shapes_and_join():
    s, h = make_synthetic(n_customers=1500, seed=1)
    assert len(s) > 0 and len(h) > 0
    assert h['session_id'].isin(set(s['session_id'])).all()


def test_no_leakage_first_visit():
    """Une premiere visite ne doit avoir aucun historique (sinon = fuite)."""
    s, _ = make_synthetic(n_customers=2000, seed=1)
    cust = build_customer_features(s)
    first = cust[cust['visit_number'] == 1]
    assert (first['prior_sessions'] == 0).all()
    assert (first['prior_purchases'] == 0).all()
    assert (first['ever_purchased_before'] == 0).all()


def test_insession_features_present():
    s, h = make_synthetic(n_customers=1500, seed=1)
    ins = build_insession_features(s, h, first_n=5)
    assert 'pre_product' in ins.columns
    assert ins['pre_hits'].notna().all()
