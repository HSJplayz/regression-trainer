# regression-trainer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![CI](https://github.com/HSJplayz/regression-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/HSJplayz/regression-trainer/actions/workflows/ci.yml)
[![Streak Day 16](https://img.shields.io/badge/Streak-Day%2016-green.svg)](https://github.com/HSJplayz/streak)

Linear and polynomial regression **from scratch** — normal equation and gradient
descent, cross-validation, metrics, JSON model persistence, a CLI, and a web UI
with offline SVG charts. Only numpy (plus Flask for the web app).

Built with numpy on **Day 16** of the daily projects streak.

## Quick start

```bash
pip install -r requirements.txt

# Train a linear model on a CSV
python cli.py train data.csv -t price

# Polynomial fit + save the trained model
python cli.py train data.csv -t price -f area --degree 2 --model model.json

# Predict with a saved model (writes a 'prediction' column)
python cli.py predict model.json --data data.csv -o predictions.csv

# Compare polynomial degrees with k-fold CV
python cli.py cv data.csv -t price -f area --degrees 1,2,3 --k 5

# Self-contained demo (no data needed)
python cli.py demo --degree 2

# Web UI
python app.py   # then open http://127.0.0.1:5000
```

## What it does

| Piece | Details |
| --- | --- |
| **Fitting** | `normal` (least squares, interpretable raw slopes) or `gd` (gradient descent, auto-standardized features) |
| **Models** | multivariate `LinearModel`; univariate `PolynomialModel` (centered x-space for stability) |
| **Metrics** | RMSE, MAE, MAPE, R², and AIC/BIC |
| **Validation** | seeded `train_test_split`, shuffled k-fold `cross_val`, `evaluate_poly_degrees` |
| **Persistence** | `save`/`load` to JSON — coefficients, scaler, feature names, metrics |
| **CLI** | `train`, `predict`, `demo`, `cv` with exit codes 0/1 |
| **Web** | paste CSV → metrics + coefficient table + 3 offline SVG charts |

### Example

```
Dataset: 150 rows, 2 feature(s): temp_c, humidity
Target:  sales
Model:   linear via 'normal'
Intercept: 385.7595
  coeff[temp_c] = -7.0765
  coeff[humidity] = 2.9065
Train  R2=0.8388  RMSE=21.6577
Test   R2=0.9183  RMSE=16.0025
CV k=5: R2=0.8370 +/- 0.0640
```

## Library use

```python
import numpy as np
from regressor import LinearModel, PolynomialModel, cross_val

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([2.1, 3.9, 6.2, 8.0, 10.1])

model = LinearModel().fit(X, y, feature_names=["x"])
print(model.summary())
print(model.predict([[6.0]]))

poly = PolynomialModel(degree=2).fit(X[:, 0], y)
print(cross_val(LinearModel, X, y, k=3))
```

## Methods, honestly

- **Normal equation** solves `θ = (XᵀX)⁻¹Xᵀy` via SVD (`np.linalg.lstsq`) — exact and
  stable enough to fit raw features, so reported slopes are "per unit of x".
- **Gradient descent** standardizes features internally (raw scales can stall or
  diverge). For high-degree polynomial bases and a large learning rate it can
  still diverge — the trainer then returns the best iterate seen and flags
  `converged=False` rather than emitting garbage silently.
- **Polynomial fits** are done in centered x-space to keep the Vandermonde-like
  design matrix well conditioned; `predict` re-centers inputs the same way.

## Project layout

```
regressor.py        core models, metrics, splitting, CV, JSON persistence
cli.py              train / predict / demo / cv commands
app.py              Flask JSON API (/health, /api/train)
templates/index.html self-contained UI with inline SVG charts
test_regressor.py   35 unittest cases
```

## Future scope

- Ridge / Lasso regularization and elastic net
- Logistic regression for classification
- Multiple polynomial features (interaction terms)
- CSV/Excel input helpers and a proper feature-scaling report
- CSV export of residuals and a downloadable model card

## Tests

```bash
python -m unittest test_regressor -v
```

35 tests: metric correctness, equality with `np.linalg.lstsq`, GD↔normal
agreement, polynomial recovery, persistence round-trips, CV, CLI exit codes,
and Flask-client API tests. No network.

## License

MIT — see [LICENSE](LICENSE).