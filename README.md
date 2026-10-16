# MLops Final Project

Handwritten digit recognition API built with DevOps & MLOps practices.

## Overview
- **App**: FastAPI service predicting handwritten digits (0-9) from 64 pixel features
- **ML**: RandomForest classifier trained on the scikit-learn `digits` dataset
- **DevOps**: Git branching (`feature -> dev -> staging -> main`), GitHub Actions CI/CD, Docker images on Docker Hub
- **MLOps**: DVC data versioning (Azure remote) + MLflow experiment tracking

## Repository structure
```
app/main.py                  # FastAPI app (GET /health, POST /predict)
ml/data_pipeline.py          # export digits dataset -> data/digits.csv
ml/train.py                  # train RandomForest + log experiment in MLflow
ml/configs/                  # baseline.json / tuned.json hyperparameter configs
ml/registry/model.pkl        # current production model
ml/README_data.md            # data versioning documentation (DVC)
ml/README_experiments.md     # experiment tracking documentation (MLflow)
ml/PROVENANCE.md             # data & model provenance
ml/registry/current_model.md # current production model card
tests/                       # pytest suite (18 tests)
Dockerfile, .dockerignore    # slim container
.github/workflows/ci.yml     # CI/CD pipeline
data/digits.csv.dvc          # DVC pointer (data on Azure)
```

## Pipeline: commit -> build -> test -> staging -> prod
1. Developer commits to a `feature/*` branch
2. Opening a PR to `dev` / `staging` / `main` triggers CI: **lint** (ruff), **test** (pytest), **docker-build**
3. Merge into `staging` pushes image `liyongqin134/mlops-final:staging`
4. Merge into `main` pushes image `liyongqin134/mlops-final:latest` (production)
5. Production image serves the FastAPI app on port 8000

## Quick start (local)
```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt -r requirements-dev.txt
python ml/data_pipeline.py
python ml/train.py
uvicorn app.main:app --port 8000
```

## API
```
GET  /health                  -> {"status": "ok"}
POST /predict                 -> {"prediction": <digit>}
     body: {"pixels": [64 floats in 0..16]}
```

## Testing
```bash
python -m pytest tests -v     # 18 passed
```

## DVC & MLflow
See `ml/README_data.md` (data versions v1/v2, Azure remote) and
`ml/README_experiments.md` (baseline vs tuned runs, production decision).

## Deployment
See `DEPLOYMENT.md`.
