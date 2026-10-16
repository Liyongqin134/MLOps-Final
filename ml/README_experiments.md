# Experiment Tracking (MLflow)

## Tracking backend
Local MLflow file backend (`mlruns/`, enabled via `MLFLOW_ALLOW_FILE_STORE=true`).
View the runs with:

mlflow ui   # open http://127.0.0.1:5000

## Experiments

Two distinct experiments were tracked (one run each):

| Experiment | Run | n_estimators | max_depth | accuracy | f1_weighted |
|------------|-----|--------------|-----------|----------|-------------|
| digits-baseline | baseline_rf | 50 | None | 0.9583 | 0.9582 |
| digits-tuned | tuned_rf | 200 | 12 | 0.9611 | 0.9609 |

## What is logged per run
- `git_commit`: SHA of the training code (reproducibility)
- `dataset_version`: md5 hash of `data/digits.csv` tracked by DVC
- hyperparameters: `n_estimators`, `max_depth`
- metrics: `accuracy`, `f1_weighted`
- artifacts: trained model (`ml/registry/model.pkl`) registered via `mlflow.sklearn.log_model`

## Production decision
`tuned_rf` is the production-worthy run:
- higher accuracy (0.9611 vs 0.9583) and higher weighted F1 (0.9609 vs 0.9582)
- more trees (200) with a depth cap (12) generalizes better on the test split

## Optimized metric
`accuracy` is the primary optimized metric: the digits dataset is balanced
(1797 samples, 10 classes), so accuracy directly measures correct digit
recognition. `f1_weighted` is tracked alongside as a safeguard against
class-imbalance effects.