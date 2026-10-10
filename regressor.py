"""From-scratch regression: linear (normal equation + gradient descent) and
polynomial fits, with metrics, cross-validation, and JSON model persistence.

Depends only on numpy. Everything is validated against numpy's own least
squares solver in the test suite.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DEFAULT_SEED = 42
EPS = 1e-12


def _as_float_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        raise ValueError("expected a non-empty array")
    return arr


def metrics(y_true, y_pred, n_params: int | None = None) -> dict:
    """Regression metrics: RMSE, MAE, MAPE, R^2 (plus AIC/BIC when n_params given)."""
    y_true = _as_float_array(y_true).ravel()
    y_pred = _as_float_array(y_pred).ravel()
    if y_true.shape != y_pred.shape or y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred must be equal-length non-empty arrays")
    resid = y_true - y_pred
    n = int(y_true.shape[0])
    sse = float((resid**2).sum())
    rmse = math.sqrt(sse / n) if sse > 0 else 0.0
    mae = float(np.abs(resid).mean())
    mape_mask = np.abs(y_true) > EPS
    mape = float((np.abs(resid[mape_mask] / y_true[mape_mask]) * 100).mean()) if mape_mask.any() else None
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > EPS else (1.0 if sse <= EPS else 0.0)
    out = {
        "n": n,
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "mape": round(mape, 6) if mape is not None else None,
        "r2": round(r2, 6),
    }
    if n_params is not None:
        out["aic"] = round(float(n * math.log(sse / max(n, 1) + EPS) + 2 * n_params), 3)
        out["bic"] = round(float(n * math.log(sse / max(n, 1) + EPS) + math.log(n) * n_params), 3)
    return out


def train_test_split(X, y, test_size: float = 0.2, seed: int = DEFAULT_SEED):
    """Deterministic, seeded train/test split. Returns (X_train, X_test, y_train, y_test)."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    X = np.asarray(X)
    y = _as_float_array(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    cut = max(1, int(n * (1.0 - test_size)))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _design_matrix(X) -> np.ndarray:
    """Add an intercept column: [1 | X]."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] == 0:
        raise ValueError("cannot fit on an empty dataset")
    return np.column_stack([np.ones(X.shape[0]), X])


class LinearModel:
    """Ordinary least squares on raw features, fit via the normal equation or
    gradient descent (which standardizes features internally)."""

    def __init__(self, method: str = "normal"):
        if method not in ("normal", "gd"):
            raise ValueError("method must be 'normal' or 'gd'")
        self.method = method
        self.coefficients: np.ndarray | None = None
        self.intercept: float | None = None
        self.n_features: int | None = None
        self.feature_names: list[str] | None = None
        self.scaler: dict | None = None
        self.train_metrics: dict | None = None
        self.converged: bool | None = None

    def fit(
        self,
        X,
        y,
        *,
        standardize: bool | None = None,
        lr: float = 0.1,
        tol: float = 1e-6,
        max_iters: int = 20000,
        seed: int = DEFAULT_SEED,
        feature_names: list[str] | None = None,
    ) -> "LinearModel":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = _as_float_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] < X.shape[1] + 1:
            raise ValueError("need at least k+1 samples for k features")

        self.n_features = X.shape[1]
        if feature_names is not None:
            if len(feature_names) != self.n_features:
                raise ValueError("feature_names must match the number of columns")
            self.feature_names = list(feature_names)
        # Auto: gradient descent standardizes (required for convergence);
        # the normal equation can fit raw features with interpretable slopes.
        resolved = self.method == "gd" if standardize is None else standardize
        Xw, self.scaler = self._prepare(X, resolved)
        D = _design_matrix(Xw)

        if self.method == "normal":
            theta, *_ = np.linalg.lstsq(D, y, rcond=None)
            self.converged = True
        else:
            theta = self._gradient_descent(D, y, lr=lr, tol=tol, max_iters=max_iters, seed=seed)

        self.intercept = float(theta[0])
        self.coefficients = np.asarray(theta[1:], dtype=float)
        self.train_metrics = metrics(y, self.predict(X), n_params=self.n_features + 1)
        return self

    def _prepare(self, X: np.ndarray, standardize: bool):
        if not standardize:
            return X, None
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < EPS] = 1.0
        scaled = (X - mean) / std
        return scaled, {"mean": mean.tolist(), "std": std.tolist()}

    def _gradient_descent(self, D: np.ndarray, y: np.ndarray, *, lr, tol, max_iters, seed) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n, p = D.shape
        theta = rng.standard_normal(p) * 0.01
        best_theta, best_cost = theta.copy(), float("inf")
        prev_cost = float("inf")
        for _ in range(max_iters):
            with np.errstate(over="ignore", invalid="ignore"):
                grad = (D.T @ (D @ theta - y)) / n
                theta = theta - lr * grad
                cost = float(((D @ theta - y) ** 2).mean())
            if not np.isfinite(cost):
                # Diverged (e.g. unstable learning rate / high-degree basis).
                self.converged = False
                return best_theta
            if cost < best_cost:
                best_theta, best_cost = theta.copy(), cost
            if abs(prev_cost - cost) <= tol * max(1.0, abs(cost)):
                self.converged = True
                return theta
            prev_cost = cost
        self.converged = False
        return best_theta

    def _to_original_space(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X
        mean = np.asarray(self.scaler["mean"])
        std = np.asarray(self.scaler["std"])
        return (X - mean) / std

    def predict(self, X) -> np.ndarray:
        if self.intercept is None or self.coefficients is None:
            raise RuntimeError("model has not been fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} feature(s), got {X.shape[1]}")
        Xw = self._to_original_space(X)
        return self.intercept + Xw @ self.coefficients

    def score(self, X, y) -> float:
        return metrics(y, self.predict(X))["r2"]

    def summary(self) -> dict:
        if self.intercept is None:
            raise RuntimeError("model has not been fitted")
        return {
            "method": self.method,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "intercept": round(self.intercept, 6),
            "coefficients": [round(c, 6) for c in self.coefficients.tolist()],
            "converged": self.converged,
            "scaler": self.scaler,
            "train_metrics": self.train_metrics,
        }

    def save(self, path) -> str:
        s = self.summary()
        s["_model"] = self.__class__.__name__
        Path(path).write_text(json.dumps(s, indent=2), encoding="utf-8")
        return str(path)

    @classmethod
    def load(cls, path) -> "LinearModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(method=data.get("method", "normal"))
        model.intercept = data["intercept"]
        model.coefficients = np.asarray(data["coefficients"], dtype=float)
        model.n_features = data["n_features"]
        model.feature_names = data.get("feature_names")
        model.scaler = data.get("scaler")
        model.converged = data.get("converged")
        model.train_metrics = data.get("train_metrics")
        return model


class PolynomialModel(LinearModel):
    """Univariate polynomial regression: y = b + c1 x + c2 x^2 + ... + cd x^d.
    Fits in centered x-space for numerical stability; prediction re-centers input."""

    def __init__(self, degree: int = 2, method: str = "normal"):
        if degree < 1 or not isinstance(degree, int):
            raise ValueError("degree must be an integer >= 1")
        super().__init__(method=method)
        self.degree = degree
        self._x_mean: float | None = None
        self._x_std: float | None = None

    def fit(self, x, y, *, lr: float = 0.1, tol: float = 1e-6, max_iters: int = 20000, seed: int = DEFAULT_SEED, feature_names: list[str] | None = None) -> "PolynomialModel":
        x = _as_float_array(x)
        y = _as_float_array(y)
        if x.shape != y.shape or x.shape[0] == 0:
            raise ValueError("x and y must be equal-length non-empty arrays")
        if feature_names is not None:
            self.feature_names = list(feature_names)
        xc = x - x.mean()
        self._x_mean = float(x.mean())
        self._x_std = float(x.std()) if x.std() > EPS else 1.0
        xc = xc / self._x_std
        D = np.column_stack([np.ones_like(xc)] + [xc**p for p in range(1, self.degree + 1)])
        self.n_features = 1
        self.scaler = None
        if self.method == "normal":
            theta, *_ = np.linalg.lstsq(D, y, rcond=None)
            self.converged = True
        else:
            self.converged = None
            theta = self._gradient_descent(D, y, lr=lr, tol=tol, max_iters=max_iters, seed=seed)
        self.intercept = float(theta[0])
        self.coefficients = np.asarray(theta[1:], dtype=float)
        self.train_metrics = metrics(y, self.predict(x), n_params=self.degree + 1)
        return self

    def _to_original_space(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_std

    def predict(self, x) -> np.ndarray:
        if self.intercept is None or self.coefficients is None:
            raise RuntimeError("model has not been fitted")
        x = _as_float_array(x)
        xc = self._to_original_space(x)
        D = np.column_stack([np.ones_like(xc)] + [xc**p for p in range(1, self.degree + 1)])
        return D @ np.concatenate([[self.intercept], self.coefficients])

    def summary(self) -> dict:
        s = super().summary()
        s["_model"] = "PolynomialModel"
        s["degree"] = self.degree
        s["x_mean"] = round(self._x_mean, 6) if self._x_mean is not None else None
        s["x_std"] = round(self._x_std, 6) if self._x_std is not None else None
        return s

    @classmethod
    def load(cls, path) -> "PolynomialModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(degree=int(data.get("degree", 2)), method=data.get("method", "normal"))
        model.intercept = data["intercept"]
        model.coefficients = np.asarray(data["coefficients"], dtype=float)
        model.n_features = data.get("n_features", 1)
        model.feature_names = data.get("feature_names")
        model.scaler = data.get("scaler")
        model.converged = data.get("converged")
        model.train_metrics = data.get("train_metrics")
        model._x_mean = data.get("x_mean")
        model._x_std = data.get("x_std")
        return model


def cross_val(model_factory, X, y, k: int = 5, seed: int = DEFAULT_SEED) -> dict:
    """Stratified-ish k-fold cross-validation (shuffled). Returns per-fold R^2
    plus mean/std. model_factory: zero-arg callable returning a fresh model."""
    X = np.asarray(X, dtype=float)
    y = _as_float_array(y)
    if X.shape[0] != y.shape[0] or k < 2 or k > X.shape[0]:
        raise ValueError("k must satisfy 2 <= k <= n, and X/y must align")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    fold_sizes = np.full(k, X.shape[0] // k, dtype=int)
    fold_sizes[: X.shape[0] % k] += 1
    scores = []
    start = 0
    for size in fold_sizes:
        test_i = idx[start : start + size]
        train_i = np.concatenate([idx[:start], idx[start + size :]])
        model = model_factory()
        model.fit(X[train_i], y[train_i])
        scores.append(model.score(X[test_i], y[test_i]))
        start += size
    arr = np.asarray(scores, dtype=float)
    return {"scores": [round(float(s), 6) for s in arr], "mean": round(float(arr.mean()), 6), "std": round(float(arr.std()), 6)}


def evaluate_poly_degrees(x, y, degrees, k: int = 5, seed: int = DEFAULT_SEED) -> dict:
    """Compare polynomial degrees with cross-validated R^2 to help pick a degree."""
    out = {}
    for d in degrees:
        cv = cross_val(lambda d=d: PolynomialModel(degree=d), x, y, k=k, seed=seed)
        out[int(d)] = {"cv_r2_mean": cv["mean"], "cv_r2_std": cv["std"], "scores": cv["scores"]}
    return out