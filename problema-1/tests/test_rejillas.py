"""Contraste escalar de rejillas, empates, prefijos y procesamiento por bloques."""

import unittest
from math import sin

import numpy as np
from src.rejillas import buscar_rejillas, generar_datos, predecir


class RejillasTest(unittest.TestCase):
    def test_oraculo_escalar_y_lotes(self) -> None:
        x = np.array([0.031, 0.223, 0.449, 0.873])
        y = np.array([1, -1, 1, -1], dtype=np.int8)
        for lote in (1, 7, 1000):
            for ajuste in buscar_rejillas(x, y, [3, 6, 9], 0.1, lote):
                esperado = min(
                    (
                        sum(
                            (1 if sin((i * 0.1) * p) >= 0 else -1) != int(etiqueta)
                            for p, etiqueta in zip(x, y, strict=True)
                        ),
                        i,
                    )
                    for i in range(1, ajuste.limite * 10 + 1)
                )
                self.assertEqual(
                    (ajuste.errores, ajuste.omega), (esperado[0], esperado[1] * 0.1)
                )

    def test_refinamiento_y_desempate(self) -> None:
        x = np.array([0.1, 0.2])
        y = np.ones(2, dtype=np.int8)
        gruesa = buscar_rejillas(x, y, [3], 0.1, 4)[0]
        fina = buscar_rejillas(x, y, [3], 0.05, 3)[0]
        self.assertEqual(gruesa.omega, 0.1)
        self.assertLessEqual(fina.errores, gruesa.errores)
        self.assertEqual(predecir(1.0, np.array([0.0])).tolist(), [1])

    def test_generacion_separada_y_determinista(self) -> None:
        x, y, _ = generar_datos(20, "entrenamiento", 20)
        x2, y2, _ = generar_datos(20, "entrenamiento", 20)
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(y, y2)
        self.assertTrue(np.all((x > 0) & (x < 1)))
        self.assertFalse(np.array_equal(x, generar_datos(20, "evaluacion", 20)[0]))

    def test_entradas_invalidas(self) -> None:
        x, y = np.array([0.1]), np.array([1], dtype=np.int8)
        for paso, lote in ((0, 1), (float("nan"), 1), (0.1, 0), (0.07, 1)):
            with self.assertRaises(ValueError):
                buscar_rejillas(x, y, [3], paso, lote)
        with self.assertRaises(ValueError):
            buscar_rejillas(np.array([0.0]), y, [3], 0.1, 1)


if __name__ == "__main__":
    unittest.main()
