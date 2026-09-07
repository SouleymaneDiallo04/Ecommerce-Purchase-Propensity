"""API de scoring de propension (FastAPI).

Charge l'artefact entraine et expose un endpoint /score qui renvoie la probabilite d'achat
d'une session a partir de ses features de contexte et d'historique.

Lancement local :
  uvicorn src.api:app --reload
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
app = FastAPI(title='GA Journey Intelligence - Propensity API', version='1.0')
_cache: dict = {}


def load_artifact(name: str = 'customer'):
    if name not in _cache:
        path = MODELS_DIR / f'{name}.joblib'
        if not path.exists():
            raise HTTPException(503, f"Modele '{name}' absent. Entraine d'abord : python -m src.pipeline train")
        _cache[name] = joblib.load(path)
    return _cache[name]


class ScoreRequest(BaseModel):
    features: dict = Field(..., description='Features de la session (contexte + historique).')
    threshold: float = 0.5
    model: str = 'customer'


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/model/{name}')
def model_info(name: str = 'customer'):
    art = load_artifact(name)
    return {'model': name, 'n_features': len(art['features']), 'features': art['features'],
            'categorical': art['cat']}


@app.post('/score')
def score(req: ScoreRequest):
    art = load_artifact(req.model)
    feats, cat, cats_map, model = art['features'], art['cat'], art['categories'], art['model']
    row = {f: req.features.get(f) for f in feats}
    df = pd.DataFrame([row], columns=feats)
    for c in cat:
        vals = df[c].astype('object').where(df[c].notna(), 'other').astype(str)
        df[c] = pd.Categorical(vals, categories=cats_map[c])
    for c in [f for f in feats if f not in cat]:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    p = float(model.predict_proba(df)[:, 1][0])
    return {'propensity': round(p, 4), 'decision': bool(p >= req.threshold),
            'threshold': req.threshold, 'model': req.model}
