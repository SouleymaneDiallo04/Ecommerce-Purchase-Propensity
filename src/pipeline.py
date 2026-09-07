"""Pipeline d'entrainement en ligne de commande.

Exemples :
  python -m src.pipeline train --source synthetic
  python -m src.pipeline train --source bigquery --project ga-propension
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib

from . import attribution as A
from . import business as B
from . import data as D
from . import evaluate as E
from . import features as F
from . import model as M
from . import validation as V

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger('pipeline')


def load_sessions_hits(source: str, project: str | None = None,
                       date_min: str = '20160801', date_max: str = '20170731'):
    if source == 'synthetic':
        return D.make_synthetic()
    from google.cloud import bigquery
    client = bigquery.Client(project=project)
    sessions = D.bq_query(client, 'sessions.sql', date_min=date_min, date_max=date_max)
    hits = D.bq_query(client, 'sequences.sql', date_min=date_min, date_max=date_max)
    return sessions, hits


def train_one(name: str, df, split_date: str, out_dir: Path) -> dict:
    cat, num = F.feature_columns(df)
    train, test = M.temporal_split(df, split_date)
    train, test = M.prepare_categories(train, test, cat)
    feats = cat + num
    raw, cal, n_iter = M.fit_model(train[feats], train['converted'], cat)
    proba = cal.predict_proba(test[feats])[:, 1]
    mets = E.metrics(test['converted'], proba)
    mets.update(n_iter=n_iter, n_train=len(train), n_test=len(test))
    categories = {c: list(train[c].cat.categories) for c in cat}
    joblib.dump({'model': cal, 'features': feats, 'cat': cat, 'categories': categories},
                out_dir / f'{name}.joblib')
    business = B.expected_value_threshold(test['converted'], proba)
    business['top30'] = B.decile_targeting(test['converted'], proba, 0.3)
    return {'metrics': mets, 'importance': E.gain_importance(raw, feats), 'business': business}


def _log_mlflow(report: dict, args) -> None:
    """Trace params + metriques dans MLflow si disponible (sinon on ignore silencieusement)."""
    try:
        import mlflow
    except Exception:
        return
    try:
        mlflow.set_experiment('ga-journey-intelligence')
        with mlflow.start_run():
            mlflow.log_params({'source': args.source, 'split_date': args.split_date})
            for name in ('insession', 'customer', 'sequence'):
                if name in report and 'metrics' in report[name]:
                    for k, v in report[name]['metrics'].items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f'{name}_{k}', float(v))
    except Exception as e:
        log.warning('mlflow ignore (%s)', e)


def main():
    ap = argparse.ArgumentParser(description='Entrainement scoring de propension + attribution.')
    ap.add_argument('cmd', choices=['train'])
    ap.add_argument('--source', default='synthetic', choices=['synthetic', 'bigquery'])
    ap.add_argument('--project', default=None, help='projet GCP si --source bigquery')
    ap.add_argument('--date-min', default='20160801')
    ap.add_argument('--date-max', default='20170731')
    ap.add_argument('--split-date', default='20170401')
    ap.add_argument('--out', default='models')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log.info('chargement des donnees (%s)', args.source)
    sessions, hits = load_sessions_hits(args.source, args.project, args.date_min, args.date_max)
    log.info('sessions=%d hits=%d conv=%.4f', len(sessions), len(hits), sessions['converted'].mean())

    insess = F.build_insession_features(sessions, hits)
    cust = F.build_customer_features(sessions)

    report = {}
    log.info('modele in-session (debut de parcours)')
    report['insession'] = train_one('insession', insess, args.split_date, out)
    log.info('modele cross-session (historique visiteur)')
    report['customer'] = train_one('customer', cust, args.split_date, out)

    log.info('modele sequence (GRU sur le parcours)')
    try:
        from . import sequence_model as SQ
        report['sequence'] = SQ.train_sequence_model(sessions, hits, args.split_date, out)
    except Exception as e:  # torch absent ou autre : on continue sans bloquer
        log.warning('modele sequence ignore (%s)', e)

    log.info('attribution markov')
    report['attribution'] = A.markov_attribution(sessions)

    log.info('validation croisee temporelle')
    cat_i, num_i = F.feature_columns(insess)
    report['insession']['cv'] = V.temporal_cv(insess, cat_i, num_i, n_splits=3)
    cat_c, num_c = F.feature_columns(cust)
    report['customer']['cv'] = V.temporal_cv(cust, cat_c, num_c, n_splits=3)

    _log_mlflow(report, args)

    (out / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    log.info('ecrit %s', out / 'metrics.json')
    for k in ('insession', 'customer', 'sequence'):
        if k in report and 'metrics' in report[k]:
            m = report[k]['metrics']
            log.info('%-11s PR-AUC=%.4f ROC-AUC=%.4f top-decile=%.1f%% top30=%.1f%%',
                     k, m['pr_auc'], m['roc_auc'], m['top_decile_capture_%'], m['top30_capture_%'])


if __name__ == '__main__':
    main()
