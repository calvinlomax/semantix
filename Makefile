.PHONY: backend frontend test build levels

backend:
	.venv/bin/python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

frontend:
	npm --prefix frontend run dev

test:
	.venv/bin/python -m pytest -q
	npm --prefix frontend test

build:
	npm --prefix frontend run build

levels:
	.venv/bin/python -m backend.scripts.generate_levels
