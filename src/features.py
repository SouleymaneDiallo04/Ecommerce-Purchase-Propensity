"""Feature engineering, concu pour eviter la fuite de donnees.

Deux jeux de features :
  build_insession_features : contexte + signaux des N PREMIERS hits (avant tout achat).
  build_customer_features  : contexte + historique des sessions STRICTEMENT ANTERIEURES du visiteur.

Les totaux de session bruts (pageviews, hits, time_on_site) sont volontairement exclus :
ils sont mecaniquement gonfles par l'achat lui-meme (fuite).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTEXT_CAT = ['channelGrouping', 'source', 'medium', 'device_category', 'os', 'country', 'sub_continent']
CONTEXT_NUM = ['is_mobile', 'visit_number', 'new_visit', 'hour']
POST_INTENT = ['cart', 'checkout', 'purchase']  # pages exclues (posterieures a l'intention d'achat)


def build_insession_features(sessions: pd.DataFrame, hits: pd.DataFrame, first_n: int = 5) -> pd.DataFrame:
    """Contexte + comportement des `first_n` premiers hits (fenetre pre-conversion, sans fuite).

    On exclut les pages panier/checkout/achat : elles sont posterieures a l'intention et
    fuiteraient le label. Robuste aux libelles GA (pagePath) comme synthetiques.
    """
    early = hits[(hits['hit_index'] <= first_n) & (~hits['page_type'].isin(POST_INTENT))]
    counts = (early.pivot_table(index='session_id', columns='page_type', values='hit_index',
                                aggfunc='count', fill_value=0)
              .rename(columns=lambda c: f'pre_{c}'))
    agg = early.groupby('session_id').agg(pre_hits=('hit_index', 'count'),
                                          pre_unique_pages=('page_type', 'nunique'),
                                          pre_seconds=('seconds', 'max'))
    seq = counts.join(agg, how='outer')
    out = sessions.merge(seq, on='session_id', how='left')
    seq_cols = list(seq.columns)
    out[seq_cols] = out[seq_cols].fillna(0)
    keep = ['session_id', 'fullVisitorId', 'date'] + CONTEXT_CAT + CONTEXT_NUM + seq_cols + ['converted']
    return out[keep]


def build_customer_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """Contexte + historique du visiteur construit uniquement a partir des sessions passees.

    Cible : conversion de la session courante, predite a partir de ce qu'on savait AVANT elle.
    """
    df = sessions.copy()
    df['dt'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values(['fullVisitorId', 'dt']).reset_index(drop=True)
    g = df.groupby('fullVisitorId', sort=False)

    df['prior_sessions'] = g.cumcount()
    df['prior_purchases'] = g['converted'].cumsum() - df['converted']
    df['prior_pageviews_sum'] = g['pageviews'].cumsum() - df['pageviews']
    df['prior_pageviews_avg'] = (df['prior_pageviews_sum'] / df['prior_sessions'].replace(0, np.nan)).fillna(0)
    df['ever_purchased_before'] = (df['prior_purchases'] > 0).astype(int)
    df['days_since_last'] = g['dt'].diff().dt.days.fillna(-1)
    df['is_returning'] = (df['prior_sessions'] > 0).astype(int)

    hist = ['prior_sessions', 'prior_purchases', 'prior_pageviews_avg',
            'ever_purchased_before', 'days_since_last', 'is_returning']
    keep = ['session_id', 'fullVisitorId', 'date'] + CONTEXT_CAT + CONTEXT_NUM + hist + ['converted']
    return df[keep]


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Renvoie (categorielles, numeriques) presentes dans df, hors identifiants et cible."""
    ignore = {'session_id', 'fullVisitorId', 'date', 'converted'}
    cat = [c for c in CONTEXT_CAT if c in df.columns]
    num = [c for c in df.columns if c not in ignore and c not in cat]
    return cat, num
