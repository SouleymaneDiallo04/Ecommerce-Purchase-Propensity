.PHONY: install train train-bq test lint tune api dashboard docker compose

install:
	pip install -r requirements.txt

lint:
	ruff check src tests

tune:
	python -m src.tuning --source synthetic --trials 30

compose:
	docker compose up --build

train:
	python -m src.pipeline train --source synthetic

train-bq:
	python -m src.pipeline train --source bigquery --project $(PROJECT)

test:
	pytest tests -q

api:
	uvicorn src.api:app --reload

dashboard:
	streamlit run app/dashboard.py

docker:
	docker build -t ga-journey-intelligence .
	docker run -p 8000:8000 ga-journey-intelligence
