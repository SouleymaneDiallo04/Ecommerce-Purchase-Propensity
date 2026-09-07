"""Modele de propension : LightGBM regularise, calibration isotonic, split temporel."""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


def make_model(n_estimators: int = 1500, random_state: int = 42, **overrides) -> lgb.LGBMClassifier:
    params = dict(n_estimators=n_estimators, learning_rate=0.03, num_leaves=31,
                  min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=random_state, verbose=-1)
    params.update(overrides)
    return lgb.LGBMClassifier(**params)


def temporal_split(df: pd.DataFrame, split_date: str):
    return df[df['date'] < split_date].copy(), df[df['date'] >= split_date].copy()


def prepare_categories(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list[str], top: int = 20):
    for c in cat_cols:
        tr, te = train[c].astype(str), test[c].astype(str)
        top_vals = tr.value_counts().head(top).index
        train[c] = tr.where(tr.isin(top_vals), 'other').astype('category')
        test[c] = pd.Categorical(te.where(te.isin(top_vals), 'other'), categories=train[c].cat.categories)
    return train, test


def fit_model(X: pd.DataFrame, y, cat_cols: list[str], random_state: int = 42):
    """Entraine avec early stopping pour trouver n_iter, puis renvoie (modele brut, modele calibre, n_iter).

    - brut : refit sur tout X (pour importance / SHAP).
    - calibre : probabilites bien calibrees (isotonic, CV interne).
    """
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    probe = make_model(random_state=random_state)
    probe.fit(Xtr, ytr, categorical_feature=cat_cols, eval_set=[(Xval, yval)], eval_metric='auc',
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    n_iter = probe.best_iteration_ or 400
    raw = make_model(n_estimators=n_iter, random_state=random_state)
    raw.fit(X, y, categorical_feature=cat_cols)
    cal = CalibratedClassifierCV(make_model(n_estimators=n_iter, random_state=random_state),
                                 method='isotonic', cv=3)
    cal.fit(X, y)
    return raw, cal, n_iter
