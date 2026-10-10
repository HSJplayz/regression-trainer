"""Command-line interface for regression-trainer: train, predict, demo, cv."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

from regressor import (
    LinearModel,
    PolynomialModel,
    cross_val,
    evaluate_poly_degrees,
    metrics,
    train_test_split,
)

DEMO_ROWS = 200
DEMO_TARGET = "y"


def _read_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"'{path}' has no data rows")
    return rows


def _numeric_cols(rows: list[dict]) -> list[str]:
    cols = []
    for name in rows[0].keys():
        vals = [r.get(name) for r in rows]
        try:
            [float(v) for v in vals if v is not None and v != ""]
            cols.append(name)
        except (TypeError, ValueError):
            continue
    return cols


def _extract(rows: list[dict], target: str, features: list[str]):
    X = []
    y = []
    for r in rows:
        y.append(float(r[target]))
        X.append([float(r[f]) for f in features])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="regression-trainer",
        description="Train and apply regression models from CSV data (numpy only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python cli.py train data.csv -t price\n"
            "  python cli.py train data.csv -t price --degree 2 --model model.json\n"
            "  python cli.py train data.csv -t height -f age -f weight --cv 5\n"
            "  python cli.py predict model.json --data data.csv -o predictions.csv\n"
            "  python cli.py demo --degree 3"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fit a regression model and report metrics")
    t.add_argument("input", help="input CSV file")
    t.add_argument("-t", "--target", required=True, help="target column name")
    t.add_argument("-f", "--features", action="append", help="feature column(s); default: all other numeric columns")
    t.add_argument("--degree", type=int, default=1, help="polynomial degree (>=2 uses first feature only, default 1 = linear)")
    t.add_argument("--method", choices=["normal", "gd"], default="normal", help="fitting method (default normal)")
    t.add_argument("--test-size", type=float, default=0.2, help="held-out test fraction (default 0.2)")
    t.add_argument("--seed", type=int, default=42, help="seed for train/test split and GD (default 42)")
    t.add_argument("--cv", type=int, help="k for k-fold cross-validation R2 report")
    t.add_argument("--model", help="save the fitted model to this JSON file")
    t.add_argument("-q", "--quiet", action="store_true", help="suppress the summary")

    pr = sub.add_parser("predict", help="predict with a saved model")
    pr.add_argument("model", help="model JSON file")
    pr.add_argument("--data", required=True, help="CSV with feature columns matching the model")
    pr.add_argument("-o", "--output", help="write predictions to a CSV with a 'prediction' column")
    pr.add_argument("-q", "--quiet", action="store_true")

    d = sub.add_parser("demo", help="fit on synthetic data and show results")
    d.add_argument("--degree", type=int, default=1, help="polynomial degree (default 1)")
    d.add_argument("--method", choices=["normal", "gd"], default="normal")
    d.add_argument("--noise", type=float, default=2.0, help="noise sigma (default 2.0)")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--model", help="save the fitted model")
    d.add_argument("-q", "--quiet", action="store_true")

    cvp = sub.add_parser("cv", help="compare polynomial degrees with cross-validation")
    cvp.add_argument("input", help="input CSV file")
    cvp.add_argument("-t", "--target", required=True, help="target column name")
    cvp.add_argument("-f", "--feature", required=True, help="feature column name")
    cvp.add_argument("--degrees", default="1,2,3", help="comma-separated degrees (default 1,2,3)")
    cvp.add_argument("--k", type=int, default=5, help="fold count (default 5)")
    cvp.add_argument("--seed", type=int, default=42)
    return p


def _print_train_header(rows, features, target, degree, method) -> None:
    print(f"Dataset: {len(rows)} rows, {len(features)} feature(s):", ", ".join(features))
    print(f"Target:  {target}")
    print(f"Model:   {'linear' if degree == 1 else f'polynomial degree {degree}'} via '{method}'")


def cmd_train(args, rows):
    features = args.features or _numeric_cols(
        [dict(r) for r in rows if r.get(args.target) not in (None, "")]
    )
    features = [f for f in features if f != args.target]
    if not features:
        raise ValueError("no numeric feature columns found (use -f to pick one)")
    X, y = _extract(rows, args.target, features)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, args.test_size, args.seed)

    if args.degree > 1:
        if len(features) != 1:
            raise ValueError("polynomial mode expects exactly one feature (-f col)")
        x, target = X[:, 0], y
        model = PolynomialModel(degree=args.degree, method=args.method).fit(x, target, feature_names=features)
        x_tr = X_tr[:, 0]
        train_m = metrics(y_tr, model.predict(x_tr), n_params=args.degree + 1)
        test_m = metrics(y_te, model.predict(X_te[:, 0]), n_params=args.degree + 1)
    else:
        model = LinearModel(method=args.method).fit(X_tr, y_tr, feature_names=features)
        train_m = model.train_metrics
        test_m = metrics(y_te, model.predict(X_te), n_params=X.shape[1] + 1)

    if not args.quiet:
        _print_train_header(rows, features, args.target, args.degree, args.method)
        if args.degree == 1:
            print(f"Intercept: {model.intercept:.4f}")
            for name, c in zip(features, model.coefficients.tolist()):
                print(f"  coeff[{name}] = {c:.4f}")
        else:
            print("Coefficients (centered x-space):")
            print(f"  intercept = {model.intercept:.4f}", [round(c, 4) for c in model.coefficients.tolist()])
        print(f"Train  R2={train_m['r2']:.4f}  RMSE={train_m['rmse']:.4f}")
        print(f"Test   R2={test_m['r2']:.4f}  RMSE={test_m['rmse']:.4f}")
        if args.cv:
            factory = PolynomialModel(degree=args.degree) if args.degree > 1 else LinearModel
            cvr = cross_val(factory, X, y, k=args.cv, seed=args.seed)
            print(f"CV k={args.cv}: R2={cvr['mean']:.4f} +/- {cvr['std']:.4f}")

    if args.model:
        model.save(args.model)
        if not args.quiet:
            print(f"Saved model -> {args.model}")
    return {"model": model, "features": features, "train": train_m, "test": test_m}


def cmd_predict(args):
    data = json.loads(Path(args.model).read_text(encoding="utf-8"))
    is_poly = data.get("_model") == "PolynomialModel"
    model = PolynomialModel.load(args.model) if is_poly else LinearModel.load(args.model)
    rows = _read_csv(args.data)
    names = data.get("feature_names")
    if names:
        missing = [n for n in names if n not in rows[0]]
        if missing:
            raise ValueError(f"column(s) missing from data: {missing}")
        columns = names
    elif is_poly:
        columns = [_numeric_cols(rows)[0]]
    else:
        columns = _numeric_cols(rows)
        nf = data["n_features"]
        if nf != len(columns):
            raise ValueError(f"model expects {nf} feature column(s), found {len(columns)} in data")
    X = np.asarray([[float(r[c]) for c in columns] for r in rows])
    preds = model.predict(X)
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = columns
            writer.writerow([*header, "prediction"])
            for r, p in zip(rows, preds):
                writer.writerow([r[c] for c in header] + [round(float(p), 6)])
        if not args.quiet:
            print(f"Wrote {len(preds)} predictions -> {args.output}")
    if not args.quiet:
        print(f"Predictions: min={preds.min():.3f} max={preds.max():.3f} mean={preds.mean():.3f}")
    return 0


def cmd_demo(args):
    rng = np.random.default_rng(args.seed)
    x = np.linspace(0, 6, DEMO_ROWS)
    y = 3.0 + 2.5 * x - (0.6 * x**2 if args.degree >= 2 else 0) + rng.normal(0, args.noise, DEMO_ROWS)
    if args.degree == 1:
        model = LinearModel(method=args.method).fit(x, y)
        m = model.train_metrics
        if not args.quiet:
            print(f"Demo: y = 3 + 2.5*x + noise  (R2={m['r2']:.4f})")
            print(f"Intercept={model.intercept:.3f}  coeff={model.coefficients[0]:.3f}  (true: 3.0, 2.5)")
    else:
        model = PolynomialModel(degree=args.degree, method=args.method).fit(x, y)
        m = model.train_metrics
        if not args.quiet:
            print(f"Demo: quadratic + noise, degree {args.degree} (R2={m['r2']:.4f})")
    if args.model:
        model.save(args.model)
        if not args.quiet:
            print(f"Saved model -> {args.model}")
    return 0


def cmd_cv(args):
    rows = _read_csv(args.input)
    X, y = _extract(rows, args.target, [args.feature])
    x = X[:, 0]
    degrees = [int(d) for d in args.degrees.split(",") if d.strip()]
    res = evaluate_poly_degrees(x, y, degrees, k=args.k, seed=args.seed)
    for d in degrees:
        r = res[d]
        print(f"degree {d}: CV R2 = {r['cv_r2_mean']:.4f} +/- {r['cv_r2_std']:.4f}  scores={r['scores']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            cmd_train(args, _read_csv(args.input))
        elif args.command == "predict":
            return cmd_predict(args)
        elif args.command == "demo":
            return cmd_demo(args)
        elif args.command == "cv":
            return cmd_cv(args)
    except (ValueError, OSError, KeyError, IndexError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"error: unexpected failure: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())