"""ERM contrastado con todos los arcos definidos por muestras pequeñas."""

import unittest
from itertools import product
from math import pi

from src.arcos import Arco, ajustar_erm


class ERMTest(unittest.TestCase):
    def comprobar_minimo(self, puntos: list[float]) -> None:
        unicos = sorted(set(puntos))
        # Oráculo independiente: evaluar todos los pares de extremos geométricos.
        hipotesis = [(-1,) * len(puntos), (1,) * len(puntos)]
        for a, b in product(unicos, repeat=2):
            hipotesis.append(
                tuple(
                    1 if (a <= p <= b if a <= b else p >= a or p <= b) else -1
                    for p in puntos
                )
            )
        for etiquetas in product((-1, 1), repeat=len(puntos)):
            esperado = min(
                sum(a != b for a, b in zip(h, etiquetas, strict=True))
                for h in hipotesis
            )
            arco = ajustar_erm(puntos, etiquetas)
            obtenido = sum(
                a != b for a, b in zip(arco.predecir(puntos), etiquetas, strict=True)
            )
            with self.subTest(puntos=puntos, etiquetas=etiquetas):
                self.assertEqual(obtenido, esperado)

    def test_minimos_exhaustivos(self) -> None:
        for n in range(1, 8):
            self.comprobar_minimo([6 * (i + 1) ** 2 / (n + 1) ** 2 for i in range(n)])

    def test_repetidos_y_puntos_cercanos(self) -> None:
        self.comprobar_minimo([0, 0, 1, 2, 2])
        self.comprobar_minimo([0, 1e-15, 2e-15])

    def test_prediccion_y_extremos_cerrados(self) -> None:
        arco = Arco(3 * pi / 2, pi / 2)
        self.assertEqual(arco.predecir([0, pi / 2, pi, 3 * pi / 2]), [1, 1, -1, 1])
        self.assertEqual(Arco(1, 1).predecir([0, 1, 2]), [-1, 1, -1])
        self.assertEqual(Arco(tipo="vacio").predecir([0]), [-1])
        self.assertEqual(Arco(tipo="completo").predecir([0]), [1])

    def test_desempates_y_permutacion(self) -> None:
        self.assertEqual(ajustar_erm([], []), Arco(tipo="vacio"))
        self.assertEqual(ajustar_erm([0, 0], [-1, 1]), Arco(tipo="vacio"))
        self.assertEqual(ajustar_erm([0, 1, 2, 3], [1, -1, 1, -1]), Arco(0, 0))
        self.assertEqual(ajustar_erm([2, 0, 3, 1], [1, 1, -1, -1]), Arco(0, 0))
        self.assertEqual(ajustar_erm([1, 2], [1, 1]), Arco(tipo="completo"))


if __name__ == "__main__":
    unittest.main()
