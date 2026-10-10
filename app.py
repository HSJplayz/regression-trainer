"""Web front-end for regression-trainer (Flask). Trains on pasted CSV, renders
offline SVG charts (no CDN) from chart-ready JSON returned by /api/train."""

from __future__ import annotations

import csv
import io
import json

from flask import Flask, Response, jsonify, request, render_template

import numpy as np

from regressor import LinearModel, PolynomialModel, metrics, train_test_split

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/")
def index():
    return render_template("index.html")


def _parse_csv(text: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("CSV has no rows (or headers are missing)")
    return rows


def _numeric_cols(rows: list[dict]) -> list[str]:
    numeric = []
    for name in rows[0].keys():
        try:
            [float(r.get(name)) for r in rows]
            numeric.append(name)
        except (TypeError, ValueError):
            continue
    return numeric


def _extract(rows: list[dict], target: str, features: list[str]):
    X, y = [], []
    for r in rows:
        y.append(float(r[target]))
        X.append([float(r[f]) for f in features])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _round_list(vals, nd: int = 4):
    return [round(float(v), nd) for v in vals]


def _fit_curve(x_raw, y_raw, model) -> dict:
    order = np.argsort(x_raw)
    xs = _round_list(x_raw[order], 4)
    pred = model.predict(x_raw[order])
    return {"x": xs, "y": _round_list(y_raw[order], 4), "pred": _round_list(pred, 4)}


@app.post("/api/train")
def api_train():
    try:
        body = request.get_json(force=True, silent=True) or {}
        rows = _parse_csv(body.get("csv", ""))
        target = body.get("target")
        features = body.get("features") or []
        degree = int(body.get("degree", 1))
        method = body.get("method", "normal")
        test_size = float(body.get("test_size", 0.2))
        seed = int(body.get("seed", 42))
        want_cv = int(body.get("cv")) if body.get("cv") else None

        if target not in rows[0]:
            raise ValueError(f"target column '{target}' not found in headers: {list(rows[0].keys())}")
        if not features:
            features = [c for c in _numeric_cols(rows) if c != target]
        if not features:
            raise ValueError("no numeric feature columns found")
        missing = [f for f in features if f not in rows[0]]
        if missing:
            raise ValueError(f"feature column(s) missing: {missing}")
        if degree > 1 and len(features) != 1:
            raise ValueError("polynomial mode needs exactly one feature")

        X, y = _extract(rows, target, features)
        if X.shape[0] < 5:
            raise ValueError("need at least 5 rows to train")

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size, seed)
        if degree > 1:
            model = PolynomialModel(degree=degree, method=method).fit(X_tr[:, 0], y_tr)
            train_m = metrics(y_tr, model.predict(X_tr[:, 0]), n_params=degree + 1)
            test_m = metrics(y_te, model.predict(X_te[:, 0]), n_params=degree + 1)
            te_pred = model.predict(X_te[:, 0])
            cv = None
            if want_cv:
                from regressor import cross_val

                cv = cross_val(lambda: PolynomialModel(degree=degree), X, y, k=want_cv, seed=seed)
        else:
            model = LinearModel(method=method).fit(X_tr, y_tr)
            train_m = model.train_metrics
            test_m = metrics(y_te, model.predict(X_te), n_params=X.shape[1] + 1)
            te_pred = model.predict(X_te)
            cv = None
            if want_cv:
                from regressor import cross_val

                cv = cross_val(LinearModel, X, y, k=want_cv, seed=seed)

        coefs = [{"name": f, "value": round(float(c), 6)} for f, c in zip(features, model.coefficients.tolist())]
        result = {
            "ok": True,
            "overview": {
                "rows": int(X.shape[0]),
                "features": features,
                "degree": degree,
                "method": method,
                "mirrors_model": "PolynomialModel" if degree > 1 else "LinearModel",
            },
            "intercept": round(float(model.intercept), 6),
            "coefficients": coefs,
            "metrics_train": train_m,
            "metrics_test": test_m,
            "cv": cv,
            "chart": {
                "actual_vs_pred": {"y": _round_list(y_te.tolist()), "pred": _round_list(te_pred.tolist())},
                "residuals": _round_list((y_te - te_pred).tolist()),
                "fit": _fit_curve(X[:, 0], y, model),
            },
        }
        return jsonify(result)
    except Exception as err:
        return jsonify({"ok": False, "error": str(err)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)