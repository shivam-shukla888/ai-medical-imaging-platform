# Makefile for AI Chest X-Ray Research Platform
.PHONY: help install verify-data train evaluate test run-api run-frontend docker-build docker-up docker-down clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install        - Install production and development dependencies"
	@echo "  make verify-data    - Verify dataset integrity, shapes, and checksum"
	@echo "  make train          - Train the CNN model on PneumoniaMNIST"
	@echo "  make evaluate       - Evaluate the best checkpoint on the test split"
	@echo "  make test           - Run automated pytest suite"
	@echo "  make run-api        - Start the FastAPI backend server on port 8000"
	@echo "  make run-frontend   - Start the Streamlit research dashboard on port 8501"
	@echo "  make docker-build   - Build Docker container images"
	@echo "  make docker-up      - Launch backend and frontend containers with Docker Compose"
	@echo "  make docker-down    - Stop and remove running Docker containers"
	@echo "  make clean          - Remove temporary Python caches and build artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

verify-data:
	$(PYTHON) scripts/verify_dataset.py

train:
	$(PYTHON) scripts/run_training.py

evaluate:
	$(PYTHON) scripts/run_evaluation.py

test:
	$(PYTHON) -m pytest -v

run-api:
	$(PYTHON) -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
