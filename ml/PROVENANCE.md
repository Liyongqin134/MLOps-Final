# Provenance (Data & Model)

## Data
- **Source**: scikit-learn `digits` dataset (`sklearn.datasets.load_digits`)
  - 1797 samples, 8x8 = 64 pixel features, labels 0-9
- **Export**: `ml/data_pipeline.py` -> `data/digits.csv` (1797 rows x 65 cols)
- **Versioning**: DVC, remote `myazure` (Azure Blob, account `wut2026`, container `dvc`)

| Version | Tag | Commit | PR | Content |
|---------|-----|--------|----|---------|
| v1 | dvc-v1 | 9d202aa | #5 | raw pixel values 0-16 |
| v2 | dvc-v2 | 606f90e | #6 + #8 + #11 | global min-max normalized to [0,1], no NaN columns |

- **Reproduce**: `git switch --detach dvc-vX && dvc pull` (verified: v1 max=16.0, v2 max=1.0)

## Model
- **Algorithm**: `RandomForestClassifier(random_state=42)`, 80/20 stratified train/test split
- **Experiments**: MLflow, local file backend (`mlruns/`)

| Experiment | Run | n_estimators | max_depth | accuracy | f1_weighted |
|------------|-----|--------------|-----------|----------|-------------|
| digits-baseline | baseline_rf | 50 | None | 0.9583 | 0.9582 |
| digits-tuned | tuned_rf | 200 | 12 | 0.9611 | 0.9609 |

- **Logged per run**: `git_commit`, `dataset_version` (DVC md5), hyperparameters,
  `accuracy`, `f1_weighted`, model artifact (`mlflow.sklearn.log_model`)
- **Production model**: tuned_rf -> `ml/registry/model.pkl`

## End-to-end reproduction
```bash
git checkout main && dvc pull
python ml/data_pipeline.py
python ml/train.py --config ml/configs/tuned.json --experiment digits-tuned --run-name tuned_rf
python -m pytest tests -v
uvicorn app.main:app --port 8000
```
