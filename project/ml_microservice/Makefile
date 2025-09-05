PY=python
VENV=ENVS
APP=app.py
EXP?=churn-exp

init:
	python3 -m venv $(VENV) && . $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt

run:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

lint:
	flake8 .

format:
	ruff check --fix . || true

build:
	docker build -t mlops-microservice:latest .

train:
    MLFLOW_EXPERIMENT_NAME=churn-exp PYTHONPATH=. python src/train.py --config configs/config.yaml
	
evaluate:
    MLFLOW_EXPERIMENT_NAME=$(EXP) PYTHONPATH=. $(PY) src/evaluate.py --config configs/config.yaml

test:
    PYTHONPATH=. $(PY) -m pytest -q
