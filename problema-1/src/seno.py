"""Práctica 1(c): precisión de P3 y búsqueda del seno en rejillas finitas."""

import argparse
import csv
import json
import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from importlib.metadata import version
from math import pi, sin
from pathlib import Path
from random import Random
from time import perf_counter

SEMILLA = 221201513
RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"


def _validar_etiquetas(etiquetas: Sequence[int]) -> None:
    if not etiquetas or any(y not in (-1, 1) for y in etiquetas):
        raise ValueError("Se requiere al menos una etiqueta, todas iguales a ±1.")


def coeficiente_exacto(etiquetas: Sequence[int]) -> int:
    """Entero N tal que omega=pi*N; sin pasar por coma flotante."""
    _validar_etiquetas(etiquetas)
    return 1 + sum(((1 - y) // 2) * 10**i for i, y in enumerate(etiquetas, 1))


def construir_float64(etiquetas: Sequence[int]) -> tuple[float, float]:
    """Construye el coeficiente y la frecuencia de P3 directamente en binary64."""
    _validar_etiquetas(etiquetas)
    coeficiente = 1.0 + sum(
        float((1 - y) // 2) * 10.0**i for i, y in enumerate(etiquetas, 1)
    )
    return coeficiente, pi * coeficiente


def predecir_float64(omega: float, m: int) -> list[int]:
    """Evalúa sin(omega*x_i) sin reducir la fase simbólicamente; signo(0)=+1."""
    return [1 if sin(omega * 10.0 ** (-i)) >= 0 else -1 for i in range(1, m + 1)]


def fases_exactas(coeficiente: int, m: int) -> list[Fraction]:
    """(omega*x_i)/pi módulo 2, usando enteros y fracciones de precisión arbitraria."""
    return [Fraction(coeficiente, 10**i) % 2 for i in range(1, m + 1)]


def signo_fase(fase: Fraction) -> int:
    """Signo de sin(pi*fase) por su fase racional, sin aproximar pi ni el seno."""
    fase %= 2
    if fase in (0, 1):
        return 1  # Convención signo(0)=+1; estos casos no aparecen en P3.
    return 1 if fase < 1 else -1


@dataclass
class Resumen:
    m: int
    casos: int
    exitos_float64: int
    fraccion_completa_float64: float
    fraccion_individual_float64: float
    exitos_exactos: int
    fraccion_completa_exacta: float
    coeficientes_inexactos_float64: int


@dataclass
class Ejemplo:
    m: int
    repeticion: int
    etiquetas: str
    predicciones_float64: str
    coeficiente_exacto: str
    coeficiente_float64_hex: str
    omega_float64_hex: str
    primer_indice_erroneo: int
    fase_reducida_exacta: str
    argumento_float64_hex: str
    seno_float64: float


def _texto(etiquetas: Sequence[int]) -> str:
    return "".join("+" if y == 1 else "-" for y in etiquetas)


def guardar_csv(
    ruta: Path,
    modelo: type[Resumen] | type[Ejemplo],
    filas: Sequence[Resumen | Ejemplo],
) -> None:
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=[f.name for f in fields(modelo)])
        escritor.writeheader()
        escritor.writerows(asdict(fila) for fila in filas)


def ejecutar_precision(m_max: int, casos: int, salida: Path) -> None:
    if (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.max_exp) != (
        2,
        53,
        1024,
    ):
        raise RuntimeError("Este experimento requiere float IEEE 754 binary64.")
    inicio = perf_counter()
    resumenes: list[Resumen] = []
    ejemplos: list[Ejemplo] = []
    for m in range(1, m_max + 1):
        completos_float = completos_exactos = aciertos = inexactos = 0
        ejemplo_guardado = False
        for repeticion in range(casos):
            rng = Random(f"{SEMILLA}:p1:seno:precision:{m}:{repeticion}")
            y = [1 if rng.getrandbits(1) else -1 for _ in range(m)]
            n = coeficiente_exacto(y)
            coef_float, omega = construir_float64(y)
            pred = predecir_float64(omega, m)
            fases = fases_exactas(n, m)
            exactos = [signo_fase(fase) for fase in fases]
            if any(fase in (0, 1) for fase in fases) or exactos != y:
                raise ArithmeticError("La verificación racional no confirmó P3.")
            fallas = [i for i in range(m) if pred[i] != y[i]]
            completos_float += not fallas
            completos_exactos += exactos == y
            aciertos += m - len(fallas)
            inexactos += int(coef_float) != n
            if fallas and not ejemplo_guardado:
                j = fallas[0] + 1
                argumento = omega * 10.0 ** (-j)
                ejemplos.append(
                    Ejemplo(
                        m,
                        repeticion,
                        _texto(y),
                        _texto(pred),
                        str(n),
                        float(coef_float).hex(),
                        float(omega).hex(),
                        j,
                        str(fases[j - 1]),
                        float(argumento).hex(),
                        sin(argumento),
                    )
                )
                ejemplo_guardado = True
        resumenes.append(
            Resumen(
                m,
                casos,
                completos_float,
                completos_float / casos,
                aciertos / (m * casos),
                completos_exactos,
                completos_exactos / casos,
                inexactos,
            )
        )
        print(
            f"m={m:2d}: float64={completos_float}/{casos}; exacto={completos_exactos}/{casos}"
        )
    tablas, figuras = salida / "tablas", salida / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)
    guardar_csv(tablas / "seno-precision.csv", Resumen, resumenes)
    guardar_csv(tablas / "seno-precision-ejemplos.csv", Ejemplo, ejemplos)
    segundos_calculo = perf_counter() - inicio
    graficar_precision(resumenes, figuras / "seno-precision.png")
    primer_fallo = next((r.m for r in resumenes if r.exitos_float64 < casos), None)
    metadatos = {
        "experimento": "practica-1ci",
        "semilla_raiz": SEMILLA,
        "subflujo": "Random(f'221201513:p1:seno:precision:{m}:{repeticion}')",
        "etiquetas": "+1 si getrandbits(1)==1, -1 en otro caso",
        "m_max": m_max,
        "casos_por_m": casos,
        "vectores_comprobados": m_max * casos,
        "etiquetas_individuales": casos * m_max * (m_max + 1) // 2,
        "primer_m_con_fallo_observado": primer_fallo,
        "ruta_float64": "suma en float, omega=math.pi*S, x=10.0**(-i), math.sin(omega*x)",
        "ruta_exacta": "N entero, Fraction(N,10**i)%2; signo según fase en (0,1) o (1,2)",
        "signo_cero": 1,
        "precision_exacta": "enteros y fracciones sin redondeo; no se aproxima pi",
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "mantisa_float_bits": sys.float_info.mant_dig,
        "matplotlib": version("matplotlib"),
        "segundos_calculo_y_tablas": segundos_calculo,
        "segundos_total": perf_counter() - inicio,
    }
    (tablas / "seno-precision-metadatos.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Primer fallo observado: m={primer_fallo}; tiempo total={metadatos['segundos_total']:.3f} s"
    )


def graficar_precision(resumenes: Sequence[Resumen], destino: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(destino.parents[1] / ".cache" / "matplotlib")
    )
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figura = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(figura)
    eje = figura.subplots()
    m = [r.m for r in resumenes]
    eje.plot(
        m,
        [r.fraccion_completa_float64 for r in resumenes],
        "o-",
        label="Vectores completos, float64",
    )
    eje.plot(
        m,
        [r.fraccion_completa_exacta for r in resumenes],
        "s:",
        label="Vectores completos, Fraction",
    )
    eje.plot(
        m,
        [r.fraccion_individual_float64 for r in resumenes],
        "--",
        label="Etiquetas individuales, float64",
    )
    eje.set_xlabel("Número de puntos m")
    eje.set_ylabel("Fracción reproducida correctamente")
    eje.set_title(f"Construcción P3: {resumenes[0].casos} etiquetaciones por m")
    eje.set_ylim(-0.03, 1.06)
    eje.set_xticks(m[::2])
    eje.grid(True, alpha=0.25)
    eje.legend()
    figura.savefig(destino, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="experimento", required=True)
    precision = subparsers.add_parser("precision", help="Práctica 1(c.i)")
    precision.add_argument("--m-max", type=int, default=25)
    precision.add_argument("--casos", type=int, default=200)
    precision.add_argument("--salida", type=Path, default=RESULTADOS)
    rejillas = subparsers.add_parser("rejillas", help="Práctica 1(c.ii)")
    rejillas.add_argument("--tamanos", type=int, nargs="+", default=[20, 50, 200])
    rejillas.add_argument(
        "--limites", type=int, nargs="+", default=[30, 300, 3000, 30000]
    )
    rejillas.add_argument("--paso", type=float, default=0.005)
    rejillas.add_argument("--lote", type=int, default=4096)
    rejillas.add_argument("--evaluacion", type=int, default=10000)
    rejillas.add_argument("--sensibilidad", action="store_true")
    rejillas.add_argument("--salida", type=Path, default=RESULTADOS)
    args = parser.parse_args()
    if args.experimento == "rejillas":
        from rejillas import ejecutar

        if min(args.tamanos) < 1 or args.evaluacion < 1:
            parser.error("Tamaños y evaluación deben ser positivos.")
        ejecutar(
            sorted(set(args.tamanos)),
            sorted(set(args.limites)),
            args.paso,
            args.lote,
            args.evaluacion,
            args.sensibilidad,
            args.salida,
        )
        return
    if not 1 <= args.m_max <= 25 or args.casos < 1:
        parser.error("Se requiere m-max entre 1 y 25 y al menos un caso.")
    ejecutar_precision(args.m_max, args.casos, args.salida)


if __name__ == "__main__":
    main()
