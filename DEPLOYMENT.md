# Deployment

## Local (without Docker)
```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt -r requirements-dev.txt
python ml/data_pipeline.py
python ml/train.py
uvicorn app.main:app --port 8000
curl http://127.0.0.1:8000/health
```

## Docker
```bash
docker build -t mlops-final:local .
docker run -d --name mlops-final-app -p 127.0.0.1:8000:8000 mlops-final:local
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"pixels": [0,0,0,13,0,0,0,0, 0,0,15,16,0,0,0,0, 0,0,16,16,0,0,0,0, 0,0,0,13,0,0,0,0, 0,0,0,13,0,0,0,0, 0,0,0,13,0,0,0,0, 0,0,13,16,13,0,0,0, 0,0,0,0,0,0,0,0]}'
```

## CI/CD (GitHub Actions, `.github/workflows/ci.yml`)
- PR to `dev`/`staging`/`main` -> lint (ruff) + test (pytest) + docker-build
- push `staging` -> build & push `liyongqin134/mlops-final:staging` (staging env)
- push `main` -> build & push `liyongqin134/mlops-final:latest` (production env)

### GitHub Secrets
| Secret | Value |
|--------|-------|
| STAGING_APP_ENV | APP_ENV=staging |
| PROD_APP_ENV | APP_ENV=production |
| DOCKERHUB_USERNAME | liyongqin134 |
| DOCKERHUB_TOKEN | <Docker Hub PAT with Read/Write/Delete> |

### Images on Docker Hub
- `liyongqin134/mlops-final:staging`
- `liyongqin134/mlops-final:latest` (production)

## Environment
`.env` file (generated from secrets in CI): `APP_ENV=staging` or `APP_ENV=production`.
