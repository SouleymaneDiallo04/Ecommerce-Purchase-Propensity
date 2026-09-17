"""Genere les figures du projet a partir du run reel (BigQuery) et de models/metrics.json.

Produit dans figures/ :
  funnel.png        tunnel de conversion (sessions par etape)
  gains.png         courbe de gains cumules du modele de production (cross-session)
  calibration.png   courbe de fiabilite (probabilites calibrees)
  attribution.png   attribution Markov (removal effect) par canal
  overview.png      panneau 2x2 des quatre figures (image de portfolio)

Usage :
  py scripts/make_figures.py --project ga-propension \
     --date-min 20170201 --date-max 20170801 --split-date 20170601
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# permet `from src import ...` quand on lance `py scripts/make_figures.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data as D
from src import features as F
from src import model as M

ACCENT = '#0f3c82'
MUTED = '#5a5a5a'
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / 'figures'


def _sessions_from_bq(project, date_min, date_max):
    from google.cloud import bigquery
    client = bigquery.Client(project=project)
    return D.bq_query(client, 'sessions.sql', date_min=date_min, date_max=date_max)


def _funnel_from_bq(project, date_min, date_max):
    from google.cloud import bigquery
    client = bigquery.Client(project=project)
    q = f"""
    SELECT h.eCommerceAction.action_type AS action_type,
           COUNT(DISTINCT CONCAT(fullVisitorId,'-',CAST(visitId AS STRING))) AS sessions
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`, UNNEST(hits) h
    WHERE _TABLE_SUFFIX BETWEEN '{date_min}' AND '{date_max}'
      AND h.eCommerceAction.action_type IN ('2','3','5','6')
    GROUP BY action_type ORDER BY action_type
    """
    return client.query(q).to_dataframe()


def _test_predictions(sessions, split_date):
    cust = F.build_customer_features(sessions)
    cat, _ = F.feature_columns(cust)
    train, test = M.temporal_split(cust, split_date)
    train, test = M.prepare_categories(train, test, cat)
    saved = joblib.load(ROOT / 'models' / 'customer.joblib')
    feats = saved['features']
    proba = saved['model'].predict_proba(test[feats])[:, 1]
    return test['converted'].to_numpy(), proba


def fig_funnel(fn):
    labels = {'2': 'Vue produit', '3': 'Ajout panier', '5': 'Checkout', '6': 'Achat'}
    fn = fn.copy()
    fn['etape'] = fn['action_type'].map(labels)
    fn = fn.dropna(subset=['etape'])
    vals = fn['sessions'].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(fn['etape'], vals, color=ACCENT)
    for i, (b, v) in enumerate(zip(bars, vals)):
        pct = '' if i == 0 else f"\n({100*v/vals[0]:.0f}% de la vue produit)"
        ax.text(b.get_x() + b.get_width()/2, v, f"{int(v):,}".replace(',', ' ') + pct,
                ha='center', va='bottom', fontsize=8, color=MUTED)
    ax.set_title('Tunnel de conversion (sessions par etape)', color=ACCENT, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, vals.max() * 1.18)
    ax.set_ylabel('Sessions')
    fig.tight_layout()
    return fig, ax


def fig_gains(y, p, ax=None):
    order = np.argsort(-p)
    y_sorted = y[order]
    cum_buyers = np.cumsum(y_sorted) / max(y.sum(), 1)
    frac = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(frac * 100, cum_buyers * 100, color=ACCENT, lw=2.2, label='Modele de production')
    ax.plot([0, 100], [0, 100], color=MUTED, ls='--', lw=1, label='Aleatoire')
    for f in (0.1, 0.3):
        capt = cum_buyers[int(f * len(y_sorted)) - 1] * 100
        ax.scatter([f * 100], [capt], color=ACCENT, zorder=5)
        ax.annotate(f"top {int(f*100)}% cibles\n= {capt:.0f}% des acheteurs",
                    (f * 100, capt), textcoords='offset points', xytext=(8, -6),
                    fontsize=8, color=ACCENT)
    ax.set_title('Courbe de gains cumules (modele cross-session)', color=ACCENT, fontweight='bold')
    ax.set_xlabel('% de visiteurs cibles (mieux scores en premier)')
    ax.set_ylabel('% d\'acheteurs captes')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    if standalone:
        fig.tight_layout()
        return fig, ax
    return ax


def fig_calibration(y, p, ax=None):
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([0, mean_pred.max()], [0, mean_pred.max()], color=MUTED, ls='--', lw=1, label='Parfait')
    ax.plot(mean_pred, frac_pos, 'o-', color=ACCENT, lw=2, label='Modele (isotonic)')
    ax.set_title('Calibration des probabilites', color=ACCENT, fontweight='bold')
    ax.set_xlabel('Probabilite predite (moyenne par bin)')
    ax.set_ylabel('Taux d\'achat observe')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    if standalone:
        fig.tight_layout()
        return fig, ax
    return ax


def fig_attribution(metrics, ax=None):
    ch = metrics['attribution']['channels']
    items = [(k, v['removal_effect']) for k, v in ch.items() if v['removal_effect'] > 0]
    items.sort(key=lambda x: x[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(names, vals, color=ACCENT)
    ax.set_title('Attribution Markov par canal (removal effect)', color=ACCENT, fontweight='bold')
    ax.set_xlabel('Impact reel sur la conversion (removal effect)')
    ax.spines[['top', 'right']].set_visible(False)
    if standalone:
        fig.tight_layout()
        return fig, ax
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='ga-propension')
    ap.add_argument('--date-min', default='20170201')
    ap.add_argument('--date-max', default='20170801')
    ap.add_argument('--split-date', default='20170601')
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)

    print('metrics.json...')
    metrics = json.loads((ROOT / 'models' / 'metrics.json').read_text(encoding='utf-8'))
    print('sessions BigQuery...')
    sessions = _sessions_from_bq(args.project, args.date_min, args.date_max)
    print(f'  sessions={len(sessions)}')
    print('predictions test...')
    y, p = _test_predictions(sessions, args.split_date)
    print(f'  n_test={len(y)} base={y.mean():.4f}')
    # scores du modele de production, servant au simulateur de profit du dashboard
    pd.DataFrame({'y_true': y.astype(int), 'proba': np.round(p, 5)}).to_csv(
        ROOT / 'models' / 'scores_customer_test.csv', index=False)
    print('funnel BigQuery...')
    fn = _funnel_from_bq(args.project, args.date_min, args.date_max)

    # figures individuelles
    f, _ = fig_funnel(fn); f.savefig(FIG / 'funnel.png', dpi=150); plt.close(f)
    f, _ = fig_gains(y, p); f.savefig(FIG / 'gains.png', dpi=150); plt.close(f)
    f, _ = fig_calibration(y, p); f.savefig(FIG / 'calibration.png', dpi=150); plt.close(f)
    f, _ = fig_attribution(metrics); f.savefig(FIG / 'attribution.png', dpi=150); plt.close(f)

    # panneau overview 2x2 (image de portfolio)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    fig_funnel_on(axes[0, 0], fn)
    fig_gains(y, p, ax=axes[0, 1])
    fig_calibration(y, p, ax=axes[1, 0])
    fig_attribution(metrics, ax=axes[1, 1])
    fig.suptitle('E-commerce Purchase Propensity  |  donnees Google Analytics reelles',
                 color=ACCENT, fontweight='bold', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / 'overview.png', dpi=150)
    plt.close(fig)
    print('OK figures ->', FIG)


def fig_funnel_on(ax, fn):
    labels = {'2': 'Vue produit', '3': 'Ajout panier', '5': 'Checkout', '6': 'Achat'}
    fn = fn.copy()
    fn['etape'] = fn['action_type'].map(labels)
    fn = fn.dropna(subset=['etape'])
    vals = fn['sessions'].to_numpy()
    bars = ax.bar(fn['etape'], vals, color=ACCENT)
    for i, (b, v) in enumerate(zip(bars, vals)):
        pct = '' if i == 0 else f"\n{100*v/vals[0]:.0f}%"
        ax.text(b.get_x() + b.get_width()/2, v, f"{int(v):,}".replace(',', ' ') + pct,
                ha='center', va='bottom', fontsize=8, color=MUTED)
    ax.set_title('Tunnel de conversion (sessions par etape)', color=ACCENT, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, vals.max() * 1.18)
    ax.set_ylabel('Sessions')


if __name__ == '__main__':
    main()
