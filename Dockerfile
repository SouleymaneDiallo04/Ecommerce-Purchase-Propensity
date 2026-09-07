FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY models ./models

EXPOSE 8000
# ${PORT} est fourni par la plupart des hebergeurs (Render, etc.), 8000 en local.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
