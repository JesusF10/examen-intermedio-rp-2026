"""Práctica 1(b): ERM exacto de arcos y curva bajo ruido Bernoulli del 10 %."""

import argparse
import csv
import json
import os
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from math import pi, sqrt, tau
from pathlib import Path
from random import Random
from statistics import mean, stdev
from time import perf_counter

from arcos import Arco, ajustar_erm
from cota import crecimiento_exacto, margen, sauer

SEMILLA = 221201513
RUIDO = 0.1
TAMANOS = (10, 20, 50, 100, 200, 500, 1000, 2000)
OBJETIVO = Arco(3 * pi / 2, pi / 2)
RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"


@dataclass
class Registro:
    n: int
    repeticion: int
    ein: float
    eout: float
    brecha: float
    cambios_entrenamiento: int
    cambios_evaluacion: int
    segundos_erm: float
    tipo: str
    inicio: float
    fin: float


@dataclass
class Resumen:
    n: int
    repeticiones: int
    ein_medio: float
    eout_medio: float
    brecha_media: float
    error_estandar_brecha: float
    cota_p1: float
    cota_p1_sauer: float
    segundos_erm_medio: float
    fraccion_cambios_entrenamiento: float
    fraccion_cambios_evaluacion: float


def generar_muestra(
    n: int, repeticion: int, fase: str, cantidad: int
) -> tuple[list[float], list[int], int]:
    """Subflujos separados para posiciones y ruido; misma ley en ambas fases."""
    prefijo = f"{SEMILLA}:p1:aprendizaje:{n}:{repeticion}:{fase}"
    rng_puntos = Random(f"{prefijo}:puntos")
    rng_ruido = Random(f"{prefijo}:ruido")
    puntos = [tau * rng_puntos.random() for _ in range(cantidad)]
    etiquetas = OBJETIVO.predecir(puntos)
    cambios = 0
    for i in range(cantidad):
        if rng_ruido.random() < RUIDO:
            etiquetas[i] *= -1
            cambios += 1
    return puntos, etiquetas, cambios


def error(arco: Arco, puntos: Sequence[float], etiquetas: Sequence[int]) -> float:
    predicciones = arco.predecir(puntos)
    return sum(a != b for a, b in zip(predicciones, etiquetas, strict=True)) / len(
        puntos
    )


def guardar_csv(ruta: Path, filas: Sequence[Registro | Resumen]) -> None:
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(asdict(filas[0])))
        escritor.writeheader()
        escritor.writerows(asdict(fila) for fila in filas)


def ejecutar(
    tamanos: Sequence[int], repeticiones: int, evaluacion: int, salida: Path
) -> None:
    inicio = perf_counter()
    registros: list[Registro] = []
    resumenes: list[Resumen] = []
    for n in tamanos:
        actuales: list[Registro] = []
        for repeticion in range(repeticiones):
            x, y, cambios_train = generar_muestra(n, repeticion, "entrenamiento", n)
            t_erm = perf_counter()
            arco = ajustar_erm(x, y)
            tiempo_erm = perf_counter() - t_erm
            ein = error(arco, x, y)
            x_eval, y_eval, cambios_eval = generar_muestra(
                n, repeticion, "evaluacion", evaluacion
            )
            eout = error(arco, x_eval, y_eval)
            # El objetivo pertenece a H: un ERM no puede cometer más errores.
            if ein > cambios_train / n + 1e-15:
                raise ArithmeticError("El ajuste supera el error del arco objetivo.")
            actuales.append(
                Registro(
                    n,
                    repeticion,
                    ein,
                    eout,
                    eout - ein,
                    cambios_train,
                    cambios_eval,
                    tiempo_erm,
                    arco.tipo,
                    arco.inicio,
                    arco.fin,
                )
            )
        registros.extend(actuales)
        resumen = Resumen(
            n=n,
            repeticiones=repeticiones,
            ein_medio=mean(r.ein for r in actuales),
            eout_medio=mean(r.eout for r in actuales),
            brecha_media=mean(r.brecha for r in actuales),
            error_estandar_brecha=stdev(r.brecha for r in actuales)
            / sqrt(repeticiones),
            cota_p1=margen(n, crecimiento_exacto),
            cota_p1_sauer=margen(n, sauer),
            segundos_erm_medio=mean(r.segundos_erm for r in actuales),
            fraccion_cambios_entrenamiento=sum(
                r.cambios_entrenamiento for r in actuales
            )
            / (n * repeticiones),
            fraccion_cambios_evaluacion=sum(r.cambios_evaluacion for r in actuales)
            / (evaluacion * repeticiones),
        )
        resumenes.append(resumen)
        print(
            f"n={n:4d}: Ein={resumen.ein_medio:.5f}, Eout={resumen.eout_medio:.5f}, "
            f"brecha={resumen.brecha_media:.5f}, P1={resumen.cota_p1:.5f}"
        )
    tablas, figuras = salida / "tablas", salida / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)
    guardar_csv(tablas / "aprendizaje-repeticiones.csv", registros)
    guardar_csv(tablas / "aprendizaje.csv", resumenes)
    segundos_experimento = perf_counter() - inicio
    graficar(resumenes, figuras / "aprendizaje.png")
    metadatos = {
        "experimento": "practica-1b",
        "semilla_raiz": SEMILLA,
        "subflujo": "Random(f'221201513:p1:aprendizaje:{n}:{repeticion}:{fase}:{fuente}')",
        "fases": ["entrenamiento", "evaluacion"],
        "fuentes": ["puntos", "ruido"],
        "objetivo": asdict(OBJETIVO),
        "distribucion": "ángulos uniformes en [0, 2*pi)",
        "ruido": RUIDO,
        "convencion_ruido": "inversiones Bernoulli independientes en ambas fases",
        "tamanos": list(tamanos),
        "repeticiones": repeticiones,
        "datos_evaluacion_por_repeticion": evaluacion,
        "delta": 0.05,
        "cota_principal": "P1 con m(2n)=2n(2n-1)+2, sin recortar a 1",
        "erm": "exacto, bloque circular de peso máximo, O(n log n)",
        "desempate": "menos puntos distintos positivos; menor indice inicial ordenado",
        "extremos": "puntos extremos del bloque, incluidos; vacío y completo explícitos",
        "python": platform.python_version(),
        "matplotlib": version("matplotlib"),
        "segundos_experimento_y_tablas": segundos_experimento,
        "segundos_total": perf_counter() - inicio,
    }
    (tablas / "aprendizaje-metadatos.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Tiempo total: {metadatos['segundos_total']:.3f} s")


def graficar(resumenes: Sequence[Resumen], destino: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(destino.parents[1] / ".cache" / "matplotlib")
    )
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figura = Figure(figsize=(9, 8), layout="constrained")
    FigureCanvasAgg(figura)
    completo, detalle = figura.subplots(2, 1, sharex=True)
    n = [r.n for r in resumenes]
    brechas = [r.brecha_media for r in resumenes]
    errores = [r.error_estandar_brecha for r in resumenes]
    completo.plot(n, [r.cota_p1 for r in resumenes], "s--", label="P1, conteo exacto")
    completo.plot(n, [r.cota_p1_sauer for r in resumenes], ":", label="P1 + Sauer")
    completo.errorbar(n, brechas, yerr=errores, fmt="o-", label="Brecha media ± 1 EE")
    completo.set_ylabel("Brecha y cota (escala completa)")
    completo.set_title("Arcos con ruido independiente del 10 %")
    completo.legend()
    detalle.errorbar(n, brechas, yerr=errores, fmt="o-", capsize=4)
    detalle.axhline(0, color="gray", linewidth=1)
    detalle.set_ylabel("Eout − Ein (detalle, ± 1 EE)")
    detalle.set_xlabel("Tamaño de entrenamiento n (escala logarítmica)")
    for eje in (completo, detalle):
        eje.set_xscale("log")
        eje.set_xticks(n, labels=[str(valor) for valor in n])
        eje.grid(True, alpha=0.25)
    figura.savefig(destino, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tamanos", type=int, nargs="+", default=TAMANOS)
    parser.add_argument("--repeticiones", type=int, default=30)
    parser.add_argument("--evaluacion", type=int, default=10000)
    parser.add_argument("--salida", type=Path, default=RESULTADOS)
    args = parser.parse_args()
    if min(args.tamanos) < 1 or args.repeticiones < 2 or args.evaluacion < 1:
        parser.error("Tamaños y evaluación positivos; al menos dos repeticiones.")
    ejecutar(sorted(set(args.tamanos)), args.repeticiones, args.evaluacion, args.salida)


if __name__ == "__main__":
    main()
