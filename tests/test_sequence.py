from src.data import make_synthetic
from src.sequence_model import build_sequences


def test_build_sequences_shape():
    s, h = make_synthetic(n_customers=800, seed=3)
    X, y, dates, vocab = build_sequences(s, h, max_len=15)
    assert X.shape == (len(s), 15)
    assert len(y) == len(s)
    assert '<pad>' in vocab and '<unk>' in vocab


def test_sequences_exclude_post_intent():
    """Les pages panier/checkout/achat ne doivent jamais entrer dans le vocabulaire (anti-fuite)."""
    s, h = make_synthetic(n_customers=800, seed=3)
    _, _, _, vocab = build_sequences(s, h, max_len=15)
    assert all(p not in vocab for p in ('cart', 'checkout', 'purchase'))
