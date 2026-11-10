"""Diagnósticos en entrenamiento, antes y después de la evaluación de prueba."""

import hashlib
import json
import os
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .comparacion import TABLAS, construir, guardar_json, predecir
from .datos import DATOS, RESULTADOS, SEMILLA, cargar_entrenamiento
from .ordinal import ModeloOrdinal


def ejecutar() -> None:
    if (DATOS / "test-evaluado.json").exists():
        raise RuntimeError(
            "El diagnóstico para selección de variables debe preceder a prueba."
        )
    inicio = perf_counter()
    x, y, atributos = cargar_entrenamiento()
    seleccion = json.loads((TABLAS / "seleccion.json").read_text())
    nombre = seleccion["mejor_boosting_cv"]
    parametros = seleccion["modelos"][nombre]["parametros"]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA + 21)
    importancias = []
    for fold, (train, val) in enumerate(cv.split(x, y)):
        modelo = construir(nombre, parametros).fit(x[train], y[train])

        def puntuacion(estimador, datos, etiquetas):
            return -mean_absolute_error(etiquetas, predecir(estimador, nombre, datos))

        resultado = permutation_importance(
            modelo,
            x[val],
            y[val],
            scoring=puntuacion,
            n_repeats=5,
            random_state=SEMILLA + 100 + fold,
            n_jobs=1,
        )
        importancias.append(resultado.importances)
        print(f"Permutación {nombre}, fold {fold} completa", flush=True)
    aumentos = np.concatenate(importancias, axis=1)
    tabla = pd.DataFrame(
        {
            "variable": atributos,
            "aumento_mae_medio": aumentos.mean(axis=1),
            "desviacion_permutaciones": aumentos.std(axis=1, ddof=1),
        }
    )
    tabla.sort_values("aumento_mae_medio", ascending=False).to_csv(
        TABLAS / "importancia-permutacion.csv", index=False
    )
    cuatro = np.argsort(-aumentos.mean(axis=1), kind="stable")[:4]
    z = StandardScaler().fit_transform(x)
    ordinal = ModeloOrdinal(alpha=0).fit(z, y)
    filas = [
        {
            "variable": atributos[i],
            "modelo": "ordinal",
            "umbral": "común",
            "coeficiente": float(ordinal.coef_[i]),
        }
        for i in range(len(atributos))
    ]
    convergencias = []
    for nivel in np.unique(y)[:-1]:
        binario = LogisticRegression(
            C=np.inf, solver="lbfgs", max_iter=3000, tol=1e-8
        ).fit(z, (y > nivel).astype(int))
        if int(binario.n_iter_[0]) >= 3000:
            raise RuntimeError("Logística binaria no convergió.")
        convergencias.append(
            {
                "umbral": int(nivel),
                "iteraciones": int(binario.n_iter_[0]),
                "positivos": int(np.sum(y > nivel)),
            }
        )
        for i, variable in enumerate(atributos):
            filas.append(
                {
                    "variable": variable,
                    "modelo": "binario",
                    "umbral": str(int(nivel)),
                    "coeficiente": float(binario.coef_[0, i]),
                }
            )
    coeficientes = pd.DataFrame(filas)
    coeficientes.to_csv(TABLAS / "coeficientes-binarios-ordinal.csv", index=False)
    meta = {
        "mejor_boosting_cv": nombre,
        "parametros": parametros,
        "cuatro_variables": [atributos[i] for i in cuatro],
        "origen_importancia": "tres folds de entrenamiento, cinco permutaciones por fold",
        "caracter": "descriptivo tras elegir hiperparámetros; no estimación final sin sesgo",
        "modelo_ordinal_diagnostico": "sin penalizar, estandarización de todo entrenamiento",
        "binarios": "C=infinito, sin penalizar, mismo escalado; P(Y>k)",
        "convergencia_binarios": convergencias,
        "n_prueba_usados": 0,
        "segundos": perf_counter() - inicio,
    }
    guardar_json(TABLAS / "diagnostico-metadatos.json", meta)
    dibujar(tabla, coeficientes, meta["cuatro_variables"], ordinal.umbrales_)
    print(json.dumps(meta, indent=2), flush=True)


def dibujar(
    tabla: pd.DataFrame,
    coeficientes: pd.DataFrame,
    cuatro: list[str],
    theta: np.ndarray,
    sufijo: str = "",
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTADOS / ".cache" / "matplotlib"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figuras = RESULTADOS / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(8, 5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    orden = tabla.sort_values("aumento_mae_medio")
    ax.barh(
        orden.variable, orden.aumento_mae_medio, xerr=orden.desviacion_permutaciones
    )
    ax.set_xlabel("Aumento de MAE al permutar (media ± desviación)")
    ax.set_title("Importancia en validación interna de entrenamiento")
    fig.savefig(figuras / f"importancia{sufijo}.png", dpi=180)
    fig = Figure(figsize=(9, 6), layout="constrained")
    FigureCanvasAgg(fig)
    ejes = fig.subplots(2, 2)
    for ax, variable in zip(ejes.flat, cuatro, strict=True):
        filas = coeficientes[coeficientes.variable == variable]
        binarios = filas[filas.modelo == "binario"]
        ax.plot(
            binarios.umbral.astype(int),
            binarios.coeficiente,
            "o-",
            label="Logísticas binarias",
        )
        ax.axhline(
            float(filas[filas.modelo == "ordinal"].coeficiente.iloc[0]),
            linestyle="--",
            color="orange",
            label="Ordinal común",
        )
        cv = filas[filas.modelo == "ordinal_cv"]
        if not cv.empty:
            ax.axhline(
                float(cv.coeficiente.iloc[0]),
                color="red",
                linestyle="-.",
                label="Ordinal elegido por CV",
            )
        ax.set_title(variable)
        ax.set_xlabel("Corte k: calidad > k")
        ax.set_ylabel("Coeficiente estandarizado")
        ax.grid(alpha=0.2)
    ejes.flat[0].legend()
    fig.savefig(figuras / f"momios{sufijo}.png", dpi=180)
    fig = Figure(figsize=(9, 4), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    z = np.linspace(min(-10, float(theta[0]) - 1), max(11, float(theta[-1]) + 1), 1600)
    from scipy.special import expit

    densidad = expit(z) * (1 - expit(z))
    cortes = np.r_[-np.inf, theta, np.inf]
    for k in range(7):
        ax.fill_between(
            z,
            0,
            densidad,
            where=(z > cortes[k]) & (z <= cortes[k + 1]),
            alpha=0.4,
            label=f"Y={k + 3}",
        )
    ax.plot(z, densidad, color="black", lw=1)
    for valor in theta:
        ax.axvline(valor, color="gray", linestyle=":")
    ax.set_xlabel("Calidad latente Y* (x estandarizado = 0, wᵀx = 0)")
    ax.set_ylabel("Densidad logística")
    ax.set_title("Variable latente y umbrales ordinales ajustados en entrenamiento")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.savefig(figuras / "latente.png", dpi=180)


def descriptivo_final() -> None:
    """Describe el mejor boosting final usando solamente entrenamiento.

    Conserva el diagnóstico preprueba y los modelos congelados. La elección del
    modelo a describir usa el ranking guardado. Solo se ajusta y evalúa con
    entrenamiento; no se consulta prueba ni se alteran los modelos finales.
    """
    if not (DATOS / "test-evaluado.json").exists():
        raise RuntimeError("Se requiere la evaluación final ya guardada.")
    inicio = perf_counter()
    huella_antes = hashlib.sha256((DATOS / "modelos.joblib").read_bytes()).hexdigest()
    metricas = pd.read_csv(TABLAS / "prueba-metricas.csv")
    nombre = str(
        metricas[metricas.modelo.isin(["M4", "M5", "M6"])]
        .sort_values(["mae", "modelo"])
        .iloc[0]
        .modelo
    )
    seleccion = json.loads((TABLAS / "seleccion.json").read_text())
    parametros = seleccion["modelos"][nombre]["parametros"]
    x, y, atributos = cargar_entrenamiento()
    matrices = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA + 21)
    for fold, (train, val) in enumerate(cv.split(x, y)):
        modelo = construir(nombre, parametros).fit(x[train], y[train])

        def puntuar(estimador, datos, etiquetas):
            return -mean_absolute_error(etiquetas, predecir(estimador, nombre, datos))

        imp = permutation_importance(
            modelo,
            x[val],
            y[val],
            scoring=puntuar,
            n_repeats=5,
            random_state=SEMILLA + 100 + fold,
            n_jobs=1,
        )
        matrices.append(imp.importances)
    aumentos = np.concatenate(matrices, axis=1)
    tabla = pd.DataFrame(
        {
            "variable": atributos,
            "aumento_mae_medio": aumentos.mean(axis=1),
            "desviacion_permutaciones": aumentos.std(axis=1, ddof=1),
        }
    )
    tabla.sort_values("aumento_mae_medio", ascending=False).to_csv(
        TABLAS / "importancia-permutacion-final.csv", index=False
    )
    cuatro = [
        atributos[i] for i in np.argsort(-aumentos.mean(axis=1), kind="stable")[:4]
    ]
    coeficientes = pd.read_csv(TABLAS / "coeficientes-binarios-ordinal.csv")
    if "ordinal_cv" not in set(coeficientes.modelo):
        final = joblib.load(DATOS / "modelos.joblib")["M3_mediana"].named_steps[
            "modelo"
        ]
        extras = pd.DataFrame(
            [
                {
                    "variable": variable,
                    "modelo": "ordinal_cv",
                    "umbral": "común",
                    "coeficiente": float(final.coef_[i]),
                }
                for i, variable in enumerate(atributos)
            ]
        )
        coeficientes = pd.concat([coeficientes, extras], ignore_index=True)
        coeficientes.to_csv(TABLAS / "coeficientes-binarios-ordinal.csv", index=False)
    theta = np.array(
        json.loads((TABLAS / "verificacion-ordinal.json").read_text())[
            "umbrales_propios"
        ]
    )
    dibujar(tabla, coeficientes, cuatro, theta, sufijo="-final")
    if (
        huella_antes
        != hashlib.sha256((DATOS / "modelos.joblib").read_bytes()).hexdigest()
    ):
        raise RuntimeError("Los modelos finales fueron alterados.")
    meta = {
        "modelo": nombre,
        "parametros_fijados": parametros,
        "cuatro_variables": cuatro,
        "origen_importancia": "validación interna de entrenamiento, tres folds y cinco permutaciones",
        "caracter": "descriptivo posterior al ranking final; no se ajusta ningún hiperparámetro",
        "n_observaciones_prueba_usadas_aqui": 0,
        "modelos_finales_sin_cambios": True,
        "segundos": perf_counter() - inicio,
    }
    guardar_json(TABLAS / "diagnostico-final-metadatos.json", meta)
    print(json.dumps(meta, indent=2), flush=True)


def regenerar_figuras() -> None:
    """Solo lee tablas ya calculadas; no consulta observaciones ni ajusta modelos."""
    theta = np.array(
        json.loads((TABLAS / "verificacion-ordinal.json").read_text())[
            "umbrales_propios"
        ]
    )
    coef = pd.read_csv(TABLAS / "coeficientes-binarios-ordinal.csv")
    for sufijo, archivo, metadatos in (
        ("", "importancia-permutacion.csv", "diagnostico-metadatos.json"),
        (
            "-final",
            "importancia-permutacion-final.csv",
            "diagnostico-final-metadatos.json",
        ),
    ):
        tabla = pd.read_csv(TABLAS / archivo)
        cuatro = json.loads((TABLAS / metadatos).read_text())["cuatro_variables"]
        dibujar(tabla, coef, cuatro, theta, sufijo=sufijo)
    from .comparacion import dibujar_confusion

    matriz = pd.read_csv(TABLAS / "confusion.csv", index_col=0).to_numpy()
    ganador = json.loads((TABLAS / "evaluacion-final.json").read_text())["ganador_mae"]
    dibujar_confusion(matriz, ganador)


def main() -> None:
    with threadpool_limits(limits=1):
        ejecutar()


if __name__ == "__main__":
    main()
