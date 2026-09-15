.PHONY: install test lint run compose-up compose-down

install:
	python -m pip install -e '.[api,nudenet,dev]'

test:
	pytest

lint:
	ruff check .

run:
	uvicorn reel_moderation.service:app --host 0.0.0.0 --port 8088

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down
