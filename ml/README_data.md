# Dataset Versioning (DVC)

## Source
The dataset is the **scikit-learn digits** dataset (`sklearn.datasets.load_digits`),
consisting of 1797 samples of 8x8 handwritten digit images (64 pixel features,
values 0-16), labels 0-9.

## Versions

| Version | Tag | Description |
|---------|-----|-------------|
| v1 | `dvc-v1` | Raw exported digits dataset (original gray-scale pixel values 0-16), exported by `ml/data_pipeline.py`. |
| v2 | `dvc-v2` | Improved version: pixel features **global** min-max normalized to [0, 1] (`ml/normalize.py`) for stable training; constant all-zero columns stay 0 (no NaN). |

## Version <-> Commit mapping
- v1 -> tag `dvc-v1` (check with `git rev-parse dvc-v1`)
- v2 -> tag `dvc-v2` (check with `git rev-parse dvc-v2`)

## How to reproduce
dvc pull -r myazure   # restore data/digits.csv from Azure

## Remote
Azure Blob Storage, container `dvc`, account `wut2026`
(credentials in `.dvc/config.local`, never committed).
