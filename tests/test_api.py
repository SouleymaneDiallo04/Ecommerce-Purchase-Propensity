from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    assert client.get('/health').json()['status'] == 'ok'


def test_model_info():
    r = client.get('/model/customer')
    assert r.status_code == 200
    assert r.json()['n_features'] > 0


def test_score_range():
    payload = {'features': {
        'channelGrouping': 'Referral', 'source': 'Referral', 'medium': 'referral',
        'device_category': 'desktop', 'os': 'Windows', 'is_mobile': 0,
        'country': 'United States', 'sub_continent': 'Northern America',
        'visit_number': 3, 'new_visit': 0, 'hour': 20,
        'prior_sessions': 2, 'prior_purchases': 1, 'prior_pageviews_avg': 6.0,
        'ever_purchased_before': 1, 'days_since_last': 5, 'is_returning': 1}}
    r = client.post('/score', json=payload)
    assert r.status_code == 200
    p = r.json()['propensity']
    assert 0.0 <= p <= 1.0
