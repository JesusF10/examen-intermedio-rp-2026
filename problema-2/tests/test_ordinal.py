"""Gradientes, estabilidad, probabilidades, decisiones y corrección monótona."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import check_grad
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from src.comparacion import corregir_monotonia, predecir
from src.ordinal import ModeloOrdinal, inicializar, nll_grad, umbrales


class OrdinalTest(unittest.TestCase):
    def test_gradientes_datos_sinteticos(self) -> None:
        rng = np.random.default_rng(221201513)
        for k in (2, 4, 7):
            x = rng.normal(size=(35, 3))
            y = np.arange(35, dtype=np.int64) % k
            p = inicializar(y, k, 3) + rng.normal(scale=0.15, size=3 + k - 1)
            for alpha in (0.0, 0.1):
                diferencia = check_grad(
                    lambda t: nll_grad(t, x, y, k, alpha)[0],
                    lambda t: nll_grad(t, x, y, k, alpha)[1],
                    p,
                    epsilon=1e-6,
                )
                self.assertLess(diferencia, 1e-4)
            self.assertTrue(np.all(np.diff(umbrales(p[3:])) > 0))

    def test_estabilidad_logaritmica(self) -> None:
        x = np.array([[-10000.0], [10000.0], [0.0]])
        y = np.array([1, 1, 0], dtype=np.int64)
        valor, grad = nll_grad(np.array([1.0, -1.0, np.log(2.0)]), x, y, 3)
        self.assertTrue(np.isfinite(valor))
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_probabilidades_y_reglas(self) -> None:
        modelo = ModeloOrdinal()
        modelo.classes_ = np.arange(1, 6)
        p = np.array([[0.3, 0.05, 0.05, 0.25, 0.35]])
        with patch.object(modelo, "predict_proba", return_value=p):
            self.assertEqual(modelo.predict(np.zeros((1, 1))).tolist(), [4])
            modelo.regla = "moda"
            self.assertEqual(modelo.predict(np.zeros((1, 1))).tolist(), [5])
        modelo.coef_ = np.array([1.0])
        modelo.umbrales_ = np.array([-2.0, -1.0, 1.0, 2.0])
        proba = modelo.predict_proba(np.array([[-100.0], [0.0], [100.0]]))
        self.assertTrue(np.all(proba >= 0))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_pava_y_redondeo(self) -> None:
        q = np.array([[0.2, 0.8], [0.9, 0.3]])
        corregidas = corregir_monotonia(q)
        np.testing.assert_allclose(corregidas, [[0.5, 0.5], [0.9, 0.3]])
        p = -np.diff(np.c_[np.ones(2), corregidas, np.zeros(2)], axis=1)
        self.assertTrue(np.all(p >= 0))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        for valor, esperado in ((3.5, 4), (1.0, 3), (10.0, 9)):
            modelo = Pipeline(
                [("modelo", DummyRegressor(strategy="constant", constant=valor))]
            ).fit(np.zeros((2, 1)), [3, 4])
            self.assertEqual(
                predecir(modelo, "M5", np.zeros((1, 1))).tolist(), [esperado]
            )


if __name__ == "__main__":
    unittest.main()
