"""Controles exactos y de representación de la construcción P3."""

import unittest
from fractions import Fraction
from itertools import product

from src.seno import (
    coeficiente_exacto,
    construir_float64,
    fases_exactas,
    predecir_float64,
    signo_fase,
)


class PrecisionSenoTest(unittest.TestCase):
    def test_caso_personal(self) -> None:
        etiquetas = [1, -1, -1]
        self.assertEqual(coeficiente_exacto(etiquetas), 1101)
        self.assertEqual(
            fases_exactas(1101, 3),
            [Fraction(1, 10), Fraction(101, 100), Fraction(1101, 1000)],
        )
        _, omega = construir_float64(etiquetas)
        self.assertEqual(predecir_float64(omega, 3), etiquetas)

    def test_construccion_exacta_exhaustiva(self) -> None:
        for m in range(1, 11):
            for etiquetas in product((-1, 1), repeat=m):
                fases = fases_exactas(coeficiente_exacto(etiquetas), m)
                self.assertEqual([signo_fase(f) for f in fases], list(etiquetas))
                self.assertTrue(all(f not in (0, 1) for f in fases))

    def test_25_puntos_y_perdida_de_unidad(self) -> None:
        for etiquetas in (
            [1] * 25,
            [-1] * 25,
            [1] * 24 + [-1],
            [(-1) ** i for i in range(25)],
        ):
            n = coeficiente_exacto(etiquetas)
            self.assertEqual([signo_fase(f) for f in fases_exactas(n, 25)], etiquetas)
        etiquetas = [1] * 24 + [-1]
        coef_float, _ = construir_float64(etiquetas)
        self.assertNotEqual(int(coef_float), coeficiente_exacto(etiquetas))

    def test_signo_y_entradas_invalidas(self) -> None:
        self.assertEqual(signo_fase(Fraction(1, 2)), 1)
        self.assertEqual(signo_fase(Fraction(3, 2)), -1)
        self.assertEqual(signo_fase(Fraction(2)), 1)
        for etiquetas in ([], [0], [1, 2]):
            with self.assertRaises(ValueError):
                coeficiente_exacto(etiquetas)
            with self.assertRaises(ValueError):
                construir_float64(etiquetas)


if __name__ == "__main__":
    unittest.main()
