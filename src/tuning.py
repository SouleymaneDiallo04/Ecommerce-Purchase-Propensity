"""Optimisation d'hyperparametres LightGBM par Optuna.

Score maximise : PR-AUC moyen en validation croisee temporelle (pas un seul split).
Exemple : python -m src.tuning --source synthetic --trials 30
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import optuna

from . import data as D
from . import features as F
from . import validation as V

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger('tuning')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_objective(df, cat, num):
    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 900, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        }
        cv = V.temporal_cv(df, cat, num, n_splits=3, model_params=params)
        return cv.get('pr_auc_mean', 0.0)
    return objective


def main():
    ap = argparse.ArgumentParser(description='Tuning Optuna du modele de propension.')
    ap.add_argument('--source', default='synthetic', choices=['synthetic'])
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--out', default='models')
    args = ap.parse_args()

    sessions, _ = D.make_synthetic()
    cust = F.build_customer_features(sessions)
    cat, num = F.feature_columns(cust)

    study = optuna.create_study(direction='maximize')
    study.optimize(make_objective(cust, cat, num), n_trials=args.trials)

    log.info('meilleur PR-AUC (CV) : %.4f', study.best_value)
    log.info('meilleurs params : %s', study.best_params)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'best_params.json').write_text(
        json.dumps({'pr_auc_cv': round(study.best_value, 4), 'params': study.best_params},
                   indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
