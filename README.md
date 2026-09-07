# E-commerce Purchase Propensity

> Prédire quels visiteurs d'un site vont acheter — à partir de leur parcours de navigation — et
> expliquer **pourquoi**. Scoring de propension d'achat, modélisation de séquence du clickstream et
> attribution des canaux sur des données Google Analytics, le tout livré comme un service de
> production (CLI, API, dashboard, tests, CI).

![CI](https://github.com/SouleymaneDiallo04/Ecommerce-Purchase-Propensity/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Lint](https://img.shields.io/badge/lint-ruff-orange)
![Tests](https://img.shields.io/badge/tests-pytest-green)

---

## Aperçu

Les équipes e-commerce veulent savoir, le plus tôt possible, si un visiteur va convertir — pour le
prioriser, personnaliser son expérience et dépenser le budget marketing là où il rapporte. Ce projet
transforme des sessions **Google Analytics** brutes (le [jeu public GA](https://console.cloud.google.com/marketplace/product/obfuscated-ga360-data/obfuscated-ga360-data)
sur BigQuery) en :

1. un **parcours client** reconstitué (clickstream ordonné, hit par hit) ;
2. un **tunnel de conversion** (vue produit → panier → checkout → achat) ;
3. des **scores de propension d'achat** via trois modèles complémentaires, **sans fuite de données** ;
4. une **attribution des canaux** (quels points de contact génèrent vraiment les conversions) ;
5. une **API** et un **dashboard** pour servir et explorer les scores.

Toute la chaîne tourne soit sur les vraies données BigQuery, soit sur un **générateur synthétique**
intégré (aucun compte cloud requis) — c'est aussi ce qu'utilisent les tests et la CI.

## Points forts

- **Trois modèles de propension, conçus pour éviter la fuite de données**
  - `in-session` — signaux des premiers hits de la visite (fenêtre pré-intention)
  - `cross-session` — l'**historique des visites passées** du visiteur (le modèle de production)
  - `sequence` — un **GRU** sur le parcours ordonné des pages (capte les schémas de navigation)
- **Attribution Markov** (removal effect) pour classer les canaux par impact réel
- **Rigueur data science** : validation croisée temporelle, **calibration** des probabilités
  (isotonic), **seuil de décision coût/gain**, tuning **Optuna**, suivi **MLflow**
- **Ingénierie de production** : API FastAPI, dashboard Streamlit, Docker / docker-compose, suite
  `pytest`, CI GitHub Actions, linting `ruff`

## Architecture

```mermaid
flowchart LR
    GA[Google Analytics sur BigQuery] -->|SQL| F[Feature engineering]
    SYN[Générateur synthétique] --> F
    F --> M1[in-session]
    F --> M2[cross-session]
    F --> M3[séquence GRU]
    F --> ATT[Attribution Markov]
    M1 & M2 & M3 --> EVAL[CV temporelle, calibration, seuil coût/gain]
    EVAL --> API[API FastAPI /score]
    EVAL --> DASH[Dashboard Streamlit]
    ATT --> DASH
```

## Résultats (données GA réelles, fenêtre de 3 mois, taux de base ≈ 1,5 %)

| Modèle | Signal utilisé | ROC-AUC | Acheteurs captés (1er décile) |
|--------|----------------|:-------:|:-----------------------------:|
| in-session | premiers hits de la visite | 0,94 \* | 79 % |
| **cross-session (production)** | **historique des visites** | **0,89** | **60 %** |
| séquence (GRU) | clickstream ordonné | 0,95 \* | 92 % |

\* Volontairement signalé : sur ce jeu de données, les scores très élevés sont **en partie dus à
l'engagement** (un visiteur qui rebondit n'achète jamais), un signal quasi trivial. Le modèle que je
déploierais réellement est **cross-session**, dont les facteurs (géographie, canal d'acquisition,
historique) sont **actionnables et connus avant la visite**. Les métriques sont aussi validées par
**validation croisée temporelle** (moyenne ± écart-type sur des fenêtres glissantes), pas un seul
split chanceux.

## Ce qui distingue ce projet

Obtenir un modèle à 0,95 sur ces données est facile. **Savoir que 0,95 est un mensonge, c'est le
métier.** Ce dépôt montre deux traques de fuite et leur correction :

- les variables d'engagement (`pageviews`, `hits`, `time_on_site`) sont **exclues** : elles sont
  gonflées par l'achat lui-même ;
- le modèle de séquence **tronque chaque parcours au premier signal d'intention** (panier, checkout,
  pages de compte / confirmation), pour qu'aucune page post-achat ne puisse fuiter dans l'entrée.

Détecter puis supprimer la fuite — et livrer le modèle honnête et actionnable — est la compétence
centrale que démontre ce projet.

## Démarrage rapide

```bash
pip install -r requirements.txt

python -m src.pipeline train --source synthetic   # entrainement sur données synthétiques (sans cloud)
pytest tests -q                                    # suite de tests
ruff check src tests                               # linting
uvicorn src.api:app --reload                       # API de scoring
streamlit run app/dashboard.py                     # dashboard
docker compose up --build                          # API + dashboard ensemble
```

Sur les vraies données Google Analytics (projet Google Cloud gratuit) :

```bash
python -m src.pipeline train --source bigquery --project VOTRE_PROJET_GCP \
       --date-min 20170401 --date-max 20170731
```

## Structure du projet

```
src/        data · features · model · sequence_model (GRU) · evaluate · attribution
            validation (CV temporelle) · business (coût/gain) · tuning (Optuna) · pipeline · api
app/        dashboard.py (Streamlit)
sql/        requêtes BigQuery (sessions, clickstream ordonné)
tests/      pytest (données, fuite, API, rigueur)
models/     artefacts entraînés + metrics.json
Dockerfile · Dockerfile.dashboard · docker-compose.yml · render.yaml   déploiement
pyproject.toml · .pre-commit-config.yaml · Makefile · DEPLOY.md · model_card.md
```

## Déploiement

En local (`docker compose up`), un dashboard public sur Hugging Face Spaces, et l'API sur Render —
voir [`DEPLOY.md`](DEPLOY.md). Les détails et limites sont documentés dans
[`model_card.md`](model_card.md).

## Stack technique

Python · pandas · scikit-learn · LightGBM · PyTorch · Optuna · MLflow · FastAPI · Streamlit ·
Docker · BigQuery · pytest · ruff · GitHub Actions

## Auteur

**Souleymane Diallo** — élève-ingénieur en Intelligence Artificielle et Data Science.
[GitHub](https://github.com/SouleymaneDiallo04) · [Portfolio](https://jeuf-tech-portfolio.vercel.app/)
