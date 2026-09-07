# Deploiement

Trois briques : le **dashboard** (URL publique cliquable), l'**API** de scoring, et un lancement
**local** des deux via Docker Compose. Images legeres (l'API et le dashboard n'embarquent pas torch).

## A. Local, tout en un (le plus simple pour verifier)

```bash
docker compose up --build
```
- API : http://localhost:8000  (essai : `curl http://localhost:8000/health`)
- Dashboard : http://localhost:8501

## B. Dashboard en ligne (URL publique) — Hugging Face Spaces

1. Cree un Space sur https://huggingface.co/new-space, **SDK = Streamlit**.
2. Ajoute au Space ces fichiers du repo : `app.py`, le dossier `app/`, le dossier `src/`,
   le dossier `models/`, et **`requirements-app.txt` renomme en `requirements.txt`**
   (deps legeres, sans torch : build rapide).
3. HF lance automatiquement `app.py` -> tu obtiens une URL publique du type
   `https://huggingface.co/spaces/<toi>/ga-journey-intelligence`.

> Alternative equivalente : Streamlit Community Cloud (https://share.streamlit.io), en pointant
> le fichier `app/dashboard.py` et en utilisant `requirements-app.txt`.

## C. API en ligne — Render

1. Pousse le repo sur GitHub.
2. Sur https://render.com : **New > Blueprint**, pointe le repo. `render.yaml` cree un service
   web Docker (`Dockerfile`, plan free, health check `/health`).
3. Tu obtiens une URL du type `https://ga-journey-api.onrender.com`. Test :
   `POST /score` avec `{"features": {"channelGrouping": "Referral", "country": "United States",
   "visit_number": 3, "prior_purchases": 1, "ever_purchased_before": 1, "days_since_last": 5}}`.

## D. Sur OVHcloud (bonus, tres pertinent pour la candidature)

Sur une instance (VPS / Public Cloud) avec Docker installe :
```bash
git clone <repo> && cd ga-journey-intelligence
docker compose up -d --build
```
Puis exposer les ports 8000 / 8501 (reverse proxy Nginx + certificat pour une vraie URL).

## Note

Le depot contient trois jeux de dependances : `requirements.txt` (dev complet, avec torch pour le
GRU), `requirements-api.txt` (API), `requirements-app.txt` (dashboard). Les images de deploiement
utilisent les deux derniers pour rester legeres.
