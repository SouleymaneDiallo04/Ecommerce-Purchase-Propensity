"""Socle donnees : generateur synthetique riche (sessions + hits + clients multi-visites)
et chargement BigQuery. Le synthetique reproduit la structure GA (parcours ordonnes,
eCommerceAction) pour developper et tester tous les modules sans authentification.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# eCommerceAction.action_type de GA : 2=vue produit, 3=ajout panier, 5=checkout, 6=achat
ACTION = {'product': '2', 'cart': '3', 'checkout': '5', 'purchase': '6'}
CHANNELS = ['Organic Search', 'Direct', 'Referral', 'Paid Search', 'Social', 'Display', 'Affiliates']
CH_P = [0.34, 0.24, 0.12, 0.10, 0.10, 0.06, 0.04]
DEVICES = ['desktop', 'mobile', 'tablet']
DEV_P = [0.55, 0.38, 0.07]
OSS = ['Windows', 'Macintosh', 'Android', 'iOS', 'Linux', 'Chrome OS']
COUNTRIES = ['United States', 'India', 'United Kingdom', 'Canada', 'France', 'Germany', 'Brazil', 'Japan', 'Other']
SUBCONT = {'United States': 'Northern America', 'Canada': 'Northern America', 'India': 'Southern Asia',
           'United Kingdom': 'Western Europe', 'France': 'Western Europe', 'Germany': 'Western Europe',
           'Brazil': 'South America', 'Japan': 'Eastern Asia', 'Other': 'Other'}


def _medium(channel: str) -> str:
    return {'Paid Search': 'cpc', 'Direct': '(none)', 'Organic Search': 'organic'}.get(channel, 'referral')


def _build_sequence(rng, channel: str, depth: int, converted: int) -> list[str]:
    """Construit un parcours de pages ordonne, coherent avec la conversion."""
    entry = 'search' if channel in ('Organic Search', 'Paid Search') else 'home'
    seq = [entry]
    for _ in range(depth):
        seq.append(rng.choice(['category', 'product', 'search', 'info'], p=[0.4, 0.4, 0.1, 0.1]))
    if converted:
        if 'product' not in seq:
            seq.append('product')
        seq += ['cart', 'checkout', 'purchase']
    else:
        if rng.random() < 0.15:
            seq.append('cart')
            if rng.random() < 0.3:
                seq.append('checkout')
    return seq


def make_synthetic(n_customers: int = 15000, seed: int = 42):
    """Renvoie (sessions_df, hits_df). Un client (fullVisitorId) a 1..k sessions datees;
    chaque session porte un parcours ordonne de hits et un label converted."""
    rng = np.random.default_rng(seed)
    cust_device = rng.choice(DEVICES, n_customers, p=DEV_P)
    cust_country = rng.choice(COUNTRIES, n_customers)
    cust_channel = rng.choice(CHANNELS, n_customers, p=CH_P)
    base_intent = rng.normal(0, 1, n_customers)
    n_sessions = 1 + rng.poisson(1.1, n_customers)

    s_rows, h_rows = [], []
    sid = 0
    base_date = pd.Timestamp('2017-01-01')
    for c in range(n_customers):
        k = int(n_sessions[c])
        days = (rng.integers(0, 40) + np.cumsum(rng.integers(1, 25, k))) % 200
        for v in range(k):
            day = int(days[v])
            channel = cust_channel[c] if rng.random() < 0.7 else rng.choice(CHANNELS, p=CH_P)
            device = cust_device[c]
            country = cust_country[c]
            hour = int(rng.integers(0, 24))
            new_visit = 1 if v == 0 else 0
            intent = (base_intent[c] + 0.4 * (v > 0)
                      + 0.5 * (channel in ('Referral', 'Paid Search'))
                      + 0.4 * (device == 'desktop') + rng.normal(0, 0.5))
            p = 1.0 / (1.0 + np.exp(-(-4.0 + 0.9 * intent)))
            converted = int(rng.random() < p)
            depth = 1 + rng.poisson(2.0 + max(intent, 0.0))
            seq = _build_sequence(rng, channel, int(depth), converted)
            sess_id = f'{c}-{sid}'
            t = 0
            for i, pg in enumerate(seq):
                t += int(rng.integers(5, 90))
                # page_token = libelle riche de page (equivalent GA pagePathLevel1)
                h_rows.append((sess_id, f'{c}', i + 1, pg, ACTION.get(pg, '0'), pg, t))
            pageviews = sum(1 for pg in seq if pg != 'purchase')
            s_rows.append((
                f'{c}', sess_id, (base_date + pd.Timedelta(days=day)).strftime('%Y%m%d'),
                channel, channel, _medium(channel), device, rng.choice(OSS),
                int(device == 'mobile'), country, SUBCONT[country], v + 1, new_visit,
                pageviews, len(seq), t, int(len(seq) == 1), hour, converted))
            sid += 1

    sessions = pd.DataFrame(s_rows, columns=[
        'fullVisitorId', 'session_id', 'date', 'channelGrouping', 'source', 'medium',
        'device_category', 'os', 'is_mobile', 'country', 'sub_continent', 'visit_number',
        'new_visit', 'pageviews', 'hits', 'time_on_site', 'bounces', 'hour', 'converted'])
    hits = pd.DataFrame(h_rows, columns=['session_id', 'fullVisitorId', 'hit_index',
                                         'page_type', 'action_type', 'page_token', 'seconds'])
    return sessions, hits


def funnel_from_hits(hits: pd.DataFrame) -> pd.DataFrame:
    labels = {'2': 'Vue produit', '3': 'Ajout panier', '5': 'Checkout', '6': 'Achat'}
    f = (hits[hits['action_type'].isin(labels)]
         .groupby('action_type')['session_id'].nunique().rename('sessions').reset_index())
    f['etape'] = f['action_type'].map(labels)
    order = {'2': 0, '3': 1, '5': 2, '6': 3}
    f = f.sort_values('action_type', key=lambda s: s.map(order))
    return f[['etape', 'sessions']].reset_index(drop=True)


# ---------- BigQuery ----------
SQL_DIR = Path(__file__).resolve().parent.parent / 'sql'


def read_sql(name: str, **params) -> str:
    sql = (SQL_DIR / name).read_text(encoding='utf-8')
    return sql.format(**params) if params else sql


def bq_query(client, name: str, **params) -> pd.DataFrame:
    return client.query(read_sql(name, **params)).to_dataframe()
