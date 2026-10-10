# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-08

### Added
- `regressor.py` — from-scratch regression core (numpy only):
  - `LinearModel` — ordinary least squares via the normal equation (`np.linalg.lstsq`) or gradient descent; auto-standardization for GD; interpretable raw-space slopes for the normal method.
  - `PolynomialModel` — univariate polynomial fits in centered x-space for numerical stability.
  - `metrics` — RMSE, MAE, MAPE, R², and AIC/BIC when a parameter count is supplied.
  - `train_test_split`, `cross_val` (shuffled k-fold), `evaluate_poly_degrees` for degree selection.
  - JSON model persistence (`save` / `load`) including coefficients, scaler, feature names, and training metrics.
  - Gradient-descent divergence guard: keeps the best iterate and reports `converged=False` instead of overflowing silently.
- `cli.py` — `train` / `predict` / `demo` / `cv` subcommands with `--degree`, `--method`, `--cv`, `--model`, feature selection, and exit codes (0 ok, 1 error).
- `app.py` + `templates/index.html` — Flask web UI: paste CSV, pick target/features, train, and view metrics, coefficient table, and three offline SVG charts (actual-vs-predicted, residual histogram, fitted curve). No CDN.
- 35 unit tests (`test_regressor.py`) including a direct equality check against `np.linalg.lstsq`, save/load round-trips, GD-vs-normal agreement, polynomial recovery, CV, CLI exit codes, and Flask-client API tests.
- GitHub Actions CI (Python 3.11 / 3.13 matrix).

[1.0.0]: https://github.com/HSJplayz/regression-trainer/releases/tag/v1.0.0