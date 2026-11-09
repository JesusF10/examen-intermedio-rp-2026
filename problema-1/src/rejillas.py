"""Práctica 1(c.ii): búsqueda exhaustiva sobre rejillas finitas, por bloques."""

import csv
import json
import os
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from decimal import Decimal
from importlib.metadata import version
from math import isfinite, pi
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

SEMILLA = 221201513
LIMITES = (30, 300, 3000, 30000)


@dataclass
class Ajuste:
    limite: int
    omega: float
    errores: int
    candidatos_acumulados: int
    segundos_busqueda_acumulados: float


@dataclass
class Resultado:
    n: int
    limite: int
    paso: float
    omega: float
    ein: float
    eout: float
    candidatos_acumulados: int
    segundos_busqueda_acumulados: float
    cambios_entrenamiento: int
    cambios_evaluacion: int
    minimo_seno_absoluto_entrenamiento: float


def predecir(omega: float, puntos: NDArray[np.float64]) -> NDArray[np.int8]:
    """Signo del seno numérico, con signo(0)=+1."""
    if not isfinite(omega) or omega <= 0:
        raise ValueError("La frecuencia debe ser finita y positiva.")
    return np.where(np.sin(omega * puntos) >= 0, 1, -1).astype(np.int8)


def buscar_rejillas(
    puntos: NDArray[np.float64],
    etiquetas: NDArray[np.int8],
    limites: Sequence[int],
    paso: float,
    lote: int,
) -> list[Ajuste]:
    """Mínimo exacto dentro de cada rejilla; ante empate, menor frecuencia.

    Se recorre solo la rejilla mayor: las anteriores son sus prefijos. El costo
    es O(n*Amax/paso), con O(n*lote) de memoria y sin usar datos de evaluación.
    """
    if not isfinite(paso) or paso <= 0 or lote < 1:
        raise ValueError("Paso finito positivo y lote positivo requeridos.")
    if (
        puntos.ndim != 1
        or puntos.size == 0
        or etiquetas.shape != puntos.shape
        or not np.all(np.isfinite(puntos) & (puntos > 0) & (puntos < 1))
        or not np.all((etiquetas == -1) | (etiquetas == 1))
    ):
        raise ValueError("Se requieren puntos en (0,1) y una etiqueta ±1 por punto.")
    if not limites or any(a < 1 for a in limites):
        raise ValueError("Se requiere al menos un límite entero positivo.")
    cantidades: list[tuple[int, int]] = []
    for limite in sorted(set(limites)):
        cantidad = Decimal(limite) / Decimal(str(paso))
        if cantidad != cantidad.to_integral_value():
            raise ValueError("El paso debe dividir exactamente cada límite.")
        cantidades.append((limite, int(cantidad)))

    inicio = perf_counter()
    mejor_error = int(puntos.size) + 1
    mejor_indice = 1
    siguiente = 1
    positivos = etiquetas > 0
    ajustes: list[Ajuste] = []
    for limite, cantidad in cantidades:
        for primero in range(siguiente, cantidad + 1, lote):
            ultimo = min(primero + lote, cantidad + 1)
            indices = np.arange(primero, ultimo, dtype=np.int64)
            frecuencias = indices * paso
            fases = frecuencias[:, None] * puntos[None, :]
            np.sin(fases, out=fases)
            errores = np.count_nonzero((fases >= 0) != positivos, axis=1)
            local = int(np.argmin(errores))
            if int(errores[local]) < mejor_error:
                mejor_error = int(errores[local])
                mejor_indice = int(indices[local])
        ajustes.append(
            Ajuste(
                limite,
                mejor_indice * paso,
                mejor_error,
                cantidad,
                perf_counter() - inicio,
            )
        )
        siguiente = cantidad + 1
    return ajustes


def generar_datos(
    n: int, fase: str, cantidad: int
) -> tuple[NDArray[np.float64], NDArray[np.int8], int]:
    prefijo = f"{SEMILLA}:p1:seno:rejillas:{n}:{fase}"
    rng_x, rng_ruido = Random(f"{prefijo}:puntos"), Random(f"{prefijo}:ruido")
    posiciones: list[float] = []
    while len(posiciones) < cantidad:
        valor = rng_x.random()
        if valor > 0:  # random() excluye 1; se excluye también 0 explícitamente.
            posiciones.append(valor)
    puntos = np.array(posiciones, dtype=np.float64)
    etiquetas = predecir(6 * pi, puntos)
    inversiones = np.array([rng_ruido.random() < 0.15 for _ in range(cantidad)])
    etiquetas[inversiones] *= -1
    return puntos, etiquetas, int(np.count_nonzero(inversiones))


def guardar_csv(ruta: Path, filas: Sequence[Resultado]) -> None:
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=[f.name for f in fields(Resultado)]
        )
        escritor.writeheader()
        escritor.writerows(asdict(fila) for fila in filas)


def ejecutar(
    tamanos: Sequence[int],
    limites: Sequence[int],
    paso: float,
    lote: int,
    evaluacion: int,
    sensibilidad: bool,
    salida: Path,
) -> None:
    inicio = perf_counter()
    principales: list[Resultado] = []
    finos: list[Resultado] = []
    for n in tamanos:
        x, y, cambios_train = generar_datos(n, "entrenamiento", n)
        x_eval, y_eval, cambios_eval = generar_datos(n, "evaluacion", evaluacion)
        for paso_actual in [paso, paso / 2] if sensibilidad else [paso]:
            ajustes = buscar_rejillas(x, y, limites, paso_actual, lote)
            for ajuste in ajustes:
                # Evaluación posterior a la selección, sin influir en el ajuste.
                ein = float(np.mean(predecir(ajuste.omega, x) != y))
                if (
                    int(np.count_nonzero(predecir(ajuste.omega, x) != y))
                    != ajuste.errores
                ):
                    raise ArithmeticError(
                        "La hipótesis elegida no confirma el conteo de errores."
                    )
                eout = float(np.mean(predecir(ajuste.omega, x_eval) != y_eval))
                fila = Resultado(
                    n,
                    ajuste.limite,
                    paso_actual,
                    ajuste.omega,
                    ein,
                    eout,
                    ajuste.candidatos_acumulados,
                    ajuste.segundos_busqueda_acumulados,
                    cambios_train,
                    cambios_eval,
                    float(np.min(np.abs(np.sin(ajuste.omega * x)))),
                )
                (principales if paso_actual == paso else finos).append(fila)
                print(
                    f"n={n:3d}, A={ajuste.limite:5d}, paso={paso_actual:g}: "
                    f"omega={ajuste.omega:.5f}, Ein={ein:.4f}, Eout={eout:.4f}",
                    flush=True,
                )
        if sensibilidad:
            for gruesa, fina in zip(
                principales[-len(ajustes) :], finos[-len(ajustes) :], strict=True
            ):
                if fina.ein > gruesa.ein:
                    raise ArithmeticError(
                        "La rejilla refinada empeora el mínimo de entrenamiento."
                    )
    tablas, figuras = salida / "tablas", salida / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)
    guardar_csv(tablas / "seno-rejillas.csv", principales)
    if sensibilidad:
        guardar_csv(tablas / "seno-rejillas-mitad-paso.csv", finos)
    segundos_calculo = perf_counter() - inicio
    graficar(principales, figuras / "seno-rejillas.png")
    metadatos = {
        "experimento": "practica-1cii",
        "semilla_raiz": SEMILLA,
        "subflujo": "Random(f'221201513:p1:seno:rejillas:{n}:{fase}:{fuente}')",
        "fases": ["entrenamiento", "evaluacion"],
        "fuentes": ["puntos", "ruido"],
        "objetivo": "signo(sin(6*pi*x))",
        "distribucion": "uniforme en (0,1)",
        "ruido": "Bernoulli 0.15 independiente en entrenamiento y evaluación",
        "tamanos": list(tamanos),
        "limites": list(limites),
        "paso_principal": paso,
        "paso_sensibilidad": paso / 2 if sensibilidad else None,
        "lote": lote,
        "evaluacion_por_n": evaluacion,
        "mismas_muestras_para_todos_los_limites_y_pasos": True,
        "signo_cero": 1,
        "desempate": "menor omega; frecuencias crecientes",
        "alcance_optimo": "exacto en la rejilla float64; aproximado respecto a (0,A] continuo",
        "candidatos_por_n_principal": int(Decimal(max(limites)) / Decimal(str(paso))),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "matplotlib": version("matplotlib"),
        "segundos_calculo_y_tablas": segundos_calculo,
        "segundos_total": perf_counter() - inicio,
    }
    (tablas / "seno-rejillas-metadatos.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Tiempo total: {metadatos['segundos_total']:.3f} s")


def graficar(filas: Sequence[Resultado], destino: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(destino.parents[1] / ".cache" / "matplotlib")
    )
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figura = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(figura)
    eje = figura.subplots()
    for n in sorted({f.n for f in filas}):
        grupo = [f for f in filas if f.n == n]
        (linea,) = eje.plot(
            [f.limite for f in grupo],
            [f.ein for f in grupo],
            "o--",
            label=f"Ein, n={n}",
        )
        eje.plot(
            [f.limite for f in grupo],
            [f.eout for f in grupo],
            "s-",
            color=linea.get_color(),
            label=f"Eout, n={n}",
        )
    eje.axhline(0.15, color="gray", linestyle=":", label="Ruido poblacional 0.15")
    eje.set_xscale("log")
    limites = sorted({f.limite for f in filas})
    eje.set_xticks(limites, labels=[str(a) for a in limites])
    eje.set_xlabel("Límite de frecuencia A (escala logarítmica)")
    eje.set_ylabel("Error de clasificación")
    eje.set_title(f"Seno: rejilla de paso {filas[0].paso:g}, ruido del 15 %")
    eje.grid(True, alpha=0.25)
    eje.legend(ncol=2)
    figura.savefig(destino, dpi=180)
