# Current Production Model

- **File**: `ml/registry/model.pkl` (loaded by `app/main.py` at startup)
- **Algorithm**: RandomForestClassifier
- **Chosen run**: `digits-tuned / tuned_rf` (see `ml/README_experiments.md`)
- **Hyperparameters**: n_estimators=200, max_depth=12, random_state=42
- **Performance** (20% test split): accuracy = 0.9611, f1_weighted = 0.9609
- **Training data**: `data/digits.csv` v2 (DVC tag `dvc-v2`, normalized [0,1])
- **Code version**: git commit SHA logged in MLflow run (`git_commit` param)

## Load
```python
import joblib
model = joblib.load("ml/registry/model.pkl")     # relative to repo root
pred = model.predict([[... 64 features ...]])[0]
```

## Why this run
`tuned_rf` beats the baseline on both accuracy (0.9611 vs 0.9583) and
weighted F1 (0.9609 vs 0.9582): more trees (200) with a depth cap (12)
generalize better on the test split.
