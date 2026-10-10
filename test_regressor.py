"""Tests for regression-trainer (stdlib unittest, numpy + Flask only)."""

import csv
import io
import json
import tempfile
import unittest
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
from cli import build_parser, main


def noisy_line(n=100, seed=3):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    y = 3.0 + 2.5 * x + rng.normal(0, 1.0, n)
    return x.reshape(-1, 1), y


def write_csv(rows, header) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
    f.close()
    return f.name


class TestMetrics(unittest.TestCase):
    def test_known_rmse_mae(self):
        m = metrics([1, 2, 3], [2, 2, 2])
        self.assertAlmostEqual(m["rmse"], (2.0 / 3.0) ** 0.5, places=5)
        self.assertAlmostEqual(m["mae"], 2.0 / 3, places=5)
        self.assertAlmostEqual(m["n"], 3)

    def test_perfect_predictions(self):
        m = metrics([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertEqual(m["rmse"], 0.0)
        self.assertEqual(m["r2"], 1.0)
        self.assertEqual(m["mape"], 0.0)

    def test_r2_constant_target_no_crash(self):
        m = metrics([5, 5, 5], [5, 5, 4.999])
        self.assertIn("r2", m)
        self.assertIsInstance(m["r2"], float)

    def test_misaligned_raises(self):
        with self.assertRaises(ValueError):
            metrics([1, 2], [1, 2, 3])

    def test_aic_bic_with_n_params(self):
        m = metrics([1, 2, 3], [1.1, 1.9, 3.1], n_params=2)
        self.assertIn("aic", m)
        self.assertIn("bic", m)


class TestLinearModel(unittest.TestCase):
    def test_recovers_true_coefficients(self):
        X, y = noisy_line(seed=5)
        model = LinearModel().fit(X, y)
        self.assertAlmostEqual(model.intercept, 3.0, delta=0.4)
        self.assertAlmostEqual(model.coefficients[0], 2.5, delta=0.4)
        self.assertGreater(model.train_metrics["r2"], 0.9)

    def test_matches_numpy_lstsq(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        y = X @ np.array([2.0, -1.0, 0.5]) + 1.3 + rng.normal(0, 0.1, 50)
        model = LinearModel().fit(X, y)
        D = np.column_stack([np.ones(50), X])
        theta, *_ = np.linalg.lstsq(D, y, rcond=None)
        np.testing.assert_allclose([model.intercept, *model.coefficients], theta, atol=1e-8)

    def test_gradient_descent_matches_normal(self):
        X, y = noisy_line(n=200, seed=4)
        normal = LinearModel(method="normal").fit(X, y)
        gd = LinearModel(method="gd").fit(X, y, lr=0.1, max_iters=50000)
        self.assertTrue(gd.converged)
        np.testing.assert_allclose(gd.predict(X), normal.predict(X), atol=1e-2)

    def test_predict_shape_and_errors(self):
        model = LinearModel().fit(np.arange(20).reshape(-1, 1), np.arange(20) * 2.0)
        self.assertEqual(model.predict([[2.0], [3.0]]).shape, (2,))
        with self.assertRaises(ValueError):
            model.predict(np.ones((3, 2)))
        with self.assertRaises(RuntimeError):
            LinearModel().predict([1.0])

    def test_too_few_samples(self):
        with self.assertRaises(ValueError):
            LinearModel().fit(np.ones((2, 2)), [1.0, 2.0])

    def test_score_method(self):
        X, y = noisy_line()
        model = LinearModel().fit(X, y)
        self.assertGreater(model.score(X, y), 0.9)


class TestSplit(unittest.TestCase):
    def test_deterministic_and_disjoint(self):
        X, y = noisy_line(n=40)
        a = train_test_split(X, y, 0.25, seed=1)
        b = train_test_split(X, y, 0.25, seed=1)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[2], b[2])
        tr, te = set(a[0].ravel().tolist()), set(a[1].ravel().tolist())
        self.assertEqual(len(tr & te), 0)

    def test_bad_test_size(self):
        with self.assertRaises(ValueError):
            train_test_split([[1], [2]], [1, 2], test_size=0.0)


class TestPersistence(unittest.TestCase):
    def test_linear_save_load_round_trip(self):
        X, y = noisy_line()
        model = LinearModel().fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "m.json")
            model.save(path)
            loaded = LinearModel.load(path)
            np.testing.assert_allclose(loaded.predict(X), model.predict(X), atol=1e-3)
            self.assertEqual(loaded.method, model.method)

    def test_poly_save_load_round_trip(self):
        x = np.linspace(-2, 2, 50)
        y = 1 + 2 * x - 0.5 * x**2 + np.random.default_rng(1).normal(0, 0.05, 50)
        model = PolynomialModel(degree=2).fit(x, y)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "p.json")
            model.save(path)
            loaded = PolynomialModel.load(path)
            self.assertEqual(loaded.degree, 2)
            np.testing.assert_allclose(loaded.predict(x), model.predict(x), atol=1e-3)


class TestPolynomial(unittest.TestCase):
    def test_recovers_quadratic(self):
        x = np.linspace(-2, 2, 60)
        y = 1 + 2 * x - 0.5 * x**2 + np.random.default_rng(2).normal(0, 0.2, 60)
        model = PolynomialModel(degree=2).fit(x, y)
        self.assertGreater(model.train_metrics["r2"], 0.98)
        np.testing.assert_allclose(model.predict(x), y, atol=1.5)

    def test_degree_one_equals_linear(self):
        X, y = noisy_line()
        lin = LinearModel().fit(X, y)
        poly = PolynomialModel(degree=1).fit(X[:, 0], y)
        self.assertGreater(poly.train_metrics["r2"], 0.9)
        np.testing.assert_allclose(poly.predict(X[:, 0]), lin.predict(X), atol=1e-2)

    def test_invalid_degree(self):
        with self.assertRaises(ValueError):
            PolynomialModel(degree=0)
        with self.assertRaises(ValueError):
            PolynomialModel(degree=1.5)

    def test_gd_divergence_is_reported_not_silent(self):
        rng = np.random.default_rng(7)
        x = np.linspace(0, 6, 200)
        y = 3 + 2.5 * x - 0.6 * x**2 + rng.normal(0, 2, 200)
        model = PolynomialModel(degree=6, method="gd").fit(x, y, lr=0.5, max_iters=5000)
        self.assertFalse(model.converged)
        self.assertTrue(np.isfinite(model.predict(x)).all())


class TestCV(unittest.TestCase):
    def test_cross_val_scores(self):
        X, y = noisy_line(n=60)
        res = cross_val(LinearModel, X, y, k=5, seed=1)
        self.assertEqual(len(res["scores"]), 5)
        self.assertAlmostEqual(sum(res["scores"]) / 5, res["mean"], places=4)
        self.assertGreater(res["mean"], 0.5)

    def test_degree_comparison_keys_and_results(self):
        x = np.linspace(0, 5, 100)
        y = 2 + x * 3 - 0.8 * x**2 + np.random.default_rng(2).normal(0, 1, 100)
        res = evaluate_poly_degrees(x, y, [1, 2, 3], k=5, seed=1)
        self.assertEqual(set(res.keys()), {1, 2, 3})
        self.assertGreater(res[2]["cv_r2_mean"], res[1]["cv_r2_mean"])


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.csv = write_csv(
            [[y, x] for x, y in zip(noisy_line(40)[0].ravel(), noisy_line(40)[1])],
            ["target", "x"],
        )

    def test_train_linear(self):
        rc = main(["train", self.csv, "-t", "target", "-f", "x", "-q"])
        self.assertEqual(rc, 0)

    def test_train_polynomial_saves_model(self):
        with tempfile.TemporaryDirectory() as d:
            m = str(Path(d) / "m.json")
            rc = main(["train", self.csv, "-t", "target", "-f", "x", "--degree", "2", "--model", m, "-q"])
            self.assertEqual(rc, 0)
            self.assertTrue(Path(m).exists())
            data = json.loads(Path(m).read_text(encoding="utf-8"))
            self.assertEqual(data["_model"], "PolynomialModel")
            self.assertEqual(data["degree"], 2)

    def test_train_cv_exit_0(self):
        rc = main(["train", self.csv, "-t", "target", "-f", "x", "--cv", "4", "-q"])
        self.assertEqual(rc, 0)

    def test_missing_file(self):
        rc = main(["train", "nope.csv", "-t", "target", "-q"])
        self.assertEqual(rc, 1)

    def test_bad_target(self):
        rc = main(["train", self.csv, "-t", "nope", "-q"])
        self.assertEqual(rc, 1)

    def test_predict_writes_csv(self):
        with tempfile.TemporaryDirectory() as d:
            m = str(Path(d) / "m.json")
            out = str(Path(d) / "pred.csv")
            main(["train", self.csv, "-t", "target", "-f", "x", "--model", m, "-q"])
            rc = main(["predict", m, "--data", self.csv, "-o", out, "-q"])
            self.assertEqual(rc, 0)
            rows = list(csv.DictReader(open(out, encoding="utf-8")))
            self.assertEqual(len(rows), 40)
            self.assertIn("prediction", rows[0])

    def test_demo(self):
        self.assertEqual(main(["demo", "-q"]), 0)
        self.assertEqual(main(["demo", "--degree", "2", "-q"]), 0)

    def test_cv_subcommand(self):
        rc = main(["cv", self.csv, "-t", "target", "-f", "x", "--degrees", "1,2", "--k", "3"])
        self.assertEqual(rc, 0)

    def test_poly_requires_single_feature(self):
        rng = np.random.default_rng(0)
        rows = [[float(rng.normal()), float(rng.normal()), float(rng.normal())] for _ in range(30)]
        csv2 = write_csv(rows, ["a", "b", "y"])
        rc = main(["train", csv2, "-t", "y", "-f", "a", "-f", "b", "--degree", "2", "-q"])
        self.assertEqual(rc, 1)


class TestWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app

        cls.client = app.test_client()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["ok"], True)

    def test_index(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("regression-trainer", r.get_data(as_text=True))

    def test_train_valid(self):
        payload = {
            "csv": "x,y\n1,3.5\n2,6.1\n3,8.0\n4,11.5\n5,13.2\n6,16.0\n7,18.4\n8,21.0\n9,23.9\n10,26.3\n",
            "target": "y",
            "features": ["x"],
            "degree": 1,
            "method": "normal",
            "test_size": 0.2,
            "seed": 42,
        }
        r = self.client.post("/api/train", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data["ok"])
        self.assertGreater(data["metrics_test"]["r2"], 0.95)
        self.assertIn("chart", data)
        self.assertEqual(len(data["coefficients"]), 1)

    def test_train_bad_target(self):
        r = self.client.post("/api/train", json={"csv": "a,b\n1,2\n2,3\n", "target": "zzz", "features": ["a"]})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(r.get_json()["ok"])

    def test_train_polynomial_web(self):
        payload = {
            "csv": "x,y\n-2,4.2\n-1,1.0\n0,0.1\n1,1.1\n2,4.3\n3,9.2\n4,16.5\n5,25.1\n",
            "target": "y",
            "features": ["x"],
            "degree": 2,
            "method": "normal",
        }
        r = self.client.post("/api/train", json=payload)
        data = r.get_json()
        self.assertTrue(data["ok"])
        self.assertGreater(data["metrics_train"]["r2"], 0.95)


if __name__ == "__main__":
    unittest.main()