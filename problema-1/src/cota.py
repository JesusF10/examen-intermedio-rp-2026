"""Tamaño de muestra del inciso teórico 1(c), sin dependencias ni aleatoriedad."""

from collections.abc import Callable
from decimal import Decimal, localcontext
from itertools import count
from math import comb, log, sqrt
from time import perf_counter

DIMENSION_VC = 3
EPSILON = Decimal("0.1")
DELTA = Decimal("0.05")


def sauer(n: int) -> int:
    """Suma de Sauer evaluada sobre 2n puntos, con d=3."""
    return sum(comb(2 * n, i) for i in range(min(2 * n, DIMENSION_VC) + 1))


def relajacion(n: int) -> int:
    """Relajación de Sauer (2n+1)^3."""
    return (2 * n + 1) ** DIMENSION_VC


def crecimiento_exacto(n: int) -> int:
    """Función de crecimiento de arcos evaluada sobre 2n puntos."""
    return 2 * n * (2 * n - 1) + 2


def margen(n: int, crecimiento: Callable[[int], int]) -> float:
    """Margen de P1 con delta=0.05."""
    return sqrt(8 * log(4 * crecimiento(n) / float(DELTA)) / n)


def margen_decimal(n: int, crecimiento: Callable[[int], int]) -> Decimal:
    """Comprobación independiente del margen con 50 dígitos decimales."""
    with localcontext() as contexto:
        contexto.prec = 50
        argumento = Decimal(4 * crecimiento(n)) / DELTA
        return (Decimal(8) * argumento.ln() / Decimal(n)).sqrt()


def minimo_n(crecimiento: Callable[[int], int]) -> int:
    """Primer entero que cumple la cota; se revisan todos los anteriores."""
    for n in count(1):
        if margen(n, crecimiento) <= float(EPSILON):
            # Evitar que un redondeo en la frontera cambie el entero elegido.
            if not margen_decimal(n, crecimiento) <= EPSILON:
                raise ArithmeticError("La comprobación decimal no confirma el mínimo.")
            if n > 1 and not margen_decimal(n - 1, crecimiento) > EPSILON:
                raise ArithmeticError("La comprobación decimal no confirma n-1.")
            return n
    raise RuntimeError("Búsqueda sin resultado.")


def main() -> None:
    inicio = perf_counter()
    print(f"d_VC={DIMENSION_VC}, epsilon={EPSILON}, delta={DELTA}; sin aleatoriedad")
    print("Método,n,margen(n-1),margen(n),n/(10*d_VC)")
    for nombre, crecimiento in (
        ("Suma de Sauer (P1+P2)", sauer),
        ("Relajación de Sauer (P1+P2)", relajacion),
        ("Crecimiento exacto (P1+1b)", crecimiento_exacto),
    ):
        n = minimo_n(crecimiento)
        anterior = margen_decimal(n - 1, crecimiento)
        actual = margen_decimal(n, crecimiento)
        print(
            f"{nombre},{n},{anterior:.14f},{actual:.14f},{n / (10 * DIMENSION_VC):.6f}"
        )
    print(f"Tiempo de cálculo: {perf_counter() - inicio:.3f} s")


if __name__ == "__main__":
    main()
