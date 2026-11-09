"""Práctica 1(a): enumeración exhaustiva en acomodos aleatorios de la circunferencia."""

import argparse
import csv
import json
import os
import platform
from importlib.metadata import version
from itertools import product
from math import comb, tau
from pathlib import Path
from random import Random
from time import perf_counter

from arcos import realizable

SEMILLA = 221201513
RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"


def ejecutar(n_max: int, acomodos: int, salida: Path) -> None:
    """Cuenta cada etiquetación en cada acomodo; no sustituye por la fórmula."""
    inicio = perf_counter()
    filas: list[dict[str, int]] = []
    detalles: list[dict[str, int]] = []
    for n in range(1, n_max + 1):
        cuentas: list[int] = []
        for repeticion in range(acomodos):
            # Subflujos independientes, sin hash() ni semilla global mutable.
            rng = Random(f"{SEMILLA}:p1:conteo:{n}:{repeticion}")
            puntos: list[float] = []
            while len(puntos) < n:
                angulo = tau * rng.random()
                if angulo not in puntos:
                    puntos.append(angulo)
            cuenta = sum(
                realizable(puntos, etiquetas)
                for etiquetas in product((-1, 1), repeat=n)
            )
            cuentas.append(cuenta)
            detalles.append({"n": n, "acomodo": repeticion, "realizables": cuenta})
        exacto = n * (n - 1) + 2
        fila = {
            "n": n,
            "acomodos": acomodos,
            "minimo_observado": min(cuentas),
            "maximo_observado": max(cuentas),
            "exacto_arcos": exacto,
            "dos_a_n": 2**n,
            "sauer_d3": sum(comb(n, i) for i in range(min(n, 3) + 1)),
        }
        filas.append(fila)
        if not min(cuentas) == max(cuentas) == exacto:
            raise ArithmeticError(f"El conteo no coincide con la teoría para n={n}.")
        print(f"n={n:2d}: mínimo={min(cuentas):3d}, máximo={max(cuentas):3d}")

    tablas = salida / "tablas"
    figuras = salida / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)
    for nombre, datos in (("conteo.csv", filas), ("conteo-acomodos.csv", detalles)):
        with (tablas / nombre).open("w", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(datos[0]))
            escritor.writeheader()
            escritor.writerows(datos)
    tiempo_conteo = perf_counter() - inicio
    graficar(filas, figuras / "conteo.png", acomodos)

    metadatos = {
        "experimento": "practica-1a",
        "semilla_raiz": SEMILLA,
        "subflujo": "random.Random(f'{semilla_raiz}:p1:conteo:{n}:{acomodo}')",
        "generador": "random.Random, semilla de texto versión 2",
        "distribucion": "ángulos uniformes en [0, 2*pi), puntos distintos",
        "n_max": n_max,
        "acomodos_por_n": acomodos,
        "etiquetaciones_revisadas": acomodos * sum(2**n for n in range(1, n_max + 1)),
        "python": platform.python_version(),
        "matplotlib": version("matplotlib"),
        "motor_grafico": "Agg",
        "segundos_conteo_y_tablas": tiempo_conteo,
        "segundos_total": perf_counter() - inicio,
    }
    (tablas / "conteo-metadatos.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Tiempo total: {metadatos['segundos_total']:.3f} s")


def graficar(filas: list[dict[str, int]], destino: Path, acomodos: int) -> None:
    """Dibuja con Matplotlib y Agg, sin ventanas ni ejecutables externos."""
    # Solo al ejecutar: caché local ignorada por Git, sin efectos al importar.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(destino.parents[1] / ".cache" / "matplotlib")
    )
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figura = Figure(figsize=(8, 5), layout="constrained")
    FigureCanvasAgg(figura)
    eje = figura.subplots()
    n = [fila["n"] for fila in filas]
    eje.plot(n, [f["maximo_observado"] for f in filas], "o-", label="Máximo observado")
    eje.plot(n, [f["dos_a_n"] for f in filas], "--", label=r"$2^n$")
    eje.plot(n, [f["sauer_d3"] for f in filas], "s:", label="Sauer (d = 3)")
    eje.set_yscale("log", base=10)
    eje.set_xlabel("Número de puntos n")
    eje.set_ylabel("Número de etiquetaciones (escala logarítmica)")
    eje.set_title(f"Arcos: conteo exhaustivo, {acomodos} acomodos por n")
    eje.set_xticks(n)
    eje.grid(True, which="both", alpha=0.25)
    eje.legend()
    figura.savefig(destino, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-max", type=int, default=12)
    parser.add_argument("--acomodos", type=int, default=100)
    parser.add_argument("--salida", type=Path, default=RESULTADOS)
    args = parser.parse_args()
    if not 1 <= args.n_max <= 12 or args.acomodos < 1:
        parser.error("Se requiere 1 <= n-max <= 12 y acomodos >= 1.")
    ejecutar(args.n_max, args.acomodos, args.salida.resolve())


if __name__ == "__main__":
    main()
