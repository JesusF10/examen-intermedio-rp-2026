"""Verificación de realizabilidad frente a arcos geométricos explícitos."""

import unittest
from itertools import product
from math import inf, nan, tau

from src.arcos import realizable


class RealizabilidadTest(unittest.TestCase):
    def test_oraculo_geometrico(self) -> None:
        # Construir todos los arcos con extremos en los huecos, sin contar cambios.
        for n in range(1, 8):
            puntos = [tau * (i + 1) ** 2 / (n + 1) ** 2 for i in range(n)]
            huecos = [
                (puntos[i] + puntos[(i + 1) % n] + (tau if i == n - 1 else 0)) / 2 % tau
                for i in range(n)
            ]
            posibles = {(-1,) * n, (1,) * n}
            for inicio, final in product(huecos, repeat=2):
                if inicio != final:
                    posibles.add(
                        tuple(
                            1 if (p - inicio) % tau <= (final - inicio) % tau else -1
                            for p in puntos
                        )
                    )
            for etiquetas in product((-1, 1), repeat=n):
                with self.subTest(n=n, etiquetas=etiquetas):
                    self.assertEqual(
                        realizable(puntos, etiquetas), etiquetas in posibles
                    )

    def test_cruce_de_cero_y_permutacion(self) -> None:
        self.assertTrue(realizable([0.1, 1, 2, 3, 6], [1, -1, -1, -1, 1]))
        self.assertTrue(realizable([2, 6, 0.1, 1, 3], [-1, 1, 1, -1, -1]))
        self.assertFalse(realizable([0, 1, 2, 3], [1, -1, 1, -1]))

    def test_repetidos(self) -> None:
        self.assertFalse(realizable([0, 0], [-1, 1]))
        self.assertTrue(realizable([1, 0, 1, 0], [1, -1, 1, -1]))

    def test_constantes_y_muestra_vacia(self) -> None:
        self.assertTrue(realizable([], []))
        self.assertTrue(realizable([0, 1, 2], [1, 1, 1]))
        self.assertTrue(realizable([0, 1, 2], [-1, -1, -1]))

    def test_puntos_cercanos_siguen_distintos(self) -> None:
        self.assertTrue(realizable([0, 1e-15], [1, -1]))

    def test_datos_invalidos(self) -> None:
        with self.assertRaises(ValueError):
            realizable([0], [])
        with self.assertRaises(ValueError):
            realizable([0], [0])
        for angulo in (-1, tau, inf, nan):
            with self.subTest(angulo=angulo), self.assertRaises(ValueError):
                realizable([angulo], [1])


if __name__ == "__main__":
    unittest.main()
