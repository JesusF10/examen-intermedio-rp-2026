"""Selección solo en entrenamiento y evaluación final única de M1..M6."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.isotonic import isotonic_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .datos import DATOS, RESULTADOS, SEMILLA, cargar_entrenamiento, cargar_prueba
from .ordinal import Array, ModeloOrdinal

NIVELES = np.arange(3, 10, dtype=np.int64)
TABLAS = RESULTADOS / "tablas"


def corregir_monotonia(q: Array) -> Array:
    return np.vstack([isotonic_regression(fila, increasing=False) for fila in q])


class BoostingAcumulado(ClassifierMixin, BaseEstimator):
    def __init__(self, hojas: int = 7):
        self.hojas = hojas

    def fit(self, x: Array, y: np.ndarray) -> BoostingAcumulado:
        self.classes_ = np.unique(y)
        self.modelos_ = []
        for k in self.classes_[:-1]:
            modelo = HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=self.hojas,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=SEMILLA + int(k),
            ).fit(x, (y > k).astype(int))
            self.modelos_.append(modelo)
        return self

    def probabilidades_crudas(self, x: Array) -> Array:
        return np.column_stack([m.predict_proba(x)[:, 1] for m in self.modelos_])

    def predict_proba(self, x: Array) -> Array:
        q = corregir_monotonia(self.probabilidades_crudas(x))
        return -np.diff(np.c_[np.ones(len(x)), q, np.zeros(len(x))], axis=1)

    def predict(self, x: Array) -> np.ndarray:
        p = self.predict_proba(x)
        return self.classes_[np.sum(np.cumsum(p, axis=1) < 0.5, axis=1)]


def construir(nombre: str, parametros: dict) -> Pipeline:
    if nombre == "M1":
        return Pipeline([("escala", StandardScaler()), ("modelo", LinearRegression())])
    if nombre == "M2":
        return Pipeline(
            [
                ("escala", StandardScaler()),
                (
                    "modelo",
                    LogisticRegression(
                        C=parametros["C"],
                        solver="lbfgs",
                        max_iter=2500,
                        tol=1e-7,
                    ),
                ),
            ]
        )
    if nombre.startswith("M3"):
        return Pipeline(
            [
                ("escala", StandardScaler()),
                (
                    "modelo",
                    ModeloOrdinal(
                        alpha=parametros["alpha"],
                        regla="moda" if nombre.endswith("moda") else "mediana",
                    ),
                ),
            ]
        )
    comunes = dict(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=parametros["hojas"],
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEMILLA,
    )
    if nombre == "M4":
        modelo = HistGradientBoostingClassifier(**comunes)
    elif nombre == "M5":
        modelo = HistGradientBoostingRegressor(**comunes)
    elif nombre == "M6":
        modelo = BoostingAcumulado(hojas=parametros["hojas"])
    else:
        raise ValueError("Modelo desconocido.")
    return Pipeline([("modelo", modelo)])


def predecir(modelo: Pipeline, nombre: str, x: Array) -> np.ndarray:
    pred = modelo.predict(x)
    if nombre in ("M1", "M5"):
        pred = np.clip(np.floor(pred + 0.5), NIVELES[0], NIVELES[-1])
    return np.asarray(pred, dtype=np.int64)


def candidatos() -> dict[str, list[dict]]:
    return {
        "M1": [{}],
        "M2": [{"C": c} for c in (0.1, 1.0, 10.0)],
        "M3_moda": [{"alpha": a} for a in (0.0, 0.01, 0.1)],
        "M3_mediana": [{"alpha": a} for a in (0.0, 0.01, 0.1)],
        **{nombre: [{"hojas": h} for h in (7, 15)] for nombre in ("M4", "M5", "M6")},
    }


def guardar_json(ruta: Path, contenido: dict) -> None:
    ruta.write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def seleccionar() -> None:
    if (DATOS / "test-evaluado.json").exists():
        raise RuntimeError(
            "Prueba ya evaluada: no se permite reabrir selección con sus resultados."
        )
    inicio = perf_counter()
    x, y, nombres = cargar_entrenamiento()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA + 21)
    folds = list(cv.split(x, y))
    TABLAS.mkdir(parents=True, exist_ok=True)
    protocolo = {
        "metrica_principal": "MAE",
        "costos": "C(y,pred)=abs(y-pred)",
        "escenario": "apoyo a revisión humana de calidad de vino; costo lineal por nivel",
        "niveles": NIVELES.tolist(),
        "costos_matriz": np.abs(NIVELES[:, None] - NIVELES[None, :]).tolist(),
        "semilla_raiz": SEMILLA,
        "semilla_cv": SEMILLA + 21,
        "cv": 3,
        "grilla": candidatos(),
        "boosting": {
            "max_iter": 150,
            "learning_rate": 0.08,
            "l2_regularization": 1.0,
            "min_samples_leaf": 20,
            "early_stopping": False,
        },
        "M1_M5": "redondeo floor(pred+.5) y recorte a [3,9]",
        "M6": "PAVA no creciente por fila; diferencias acumuladas y mediana inferior",
        "variables_diagnostico": "cuatro mayores importancias por permutación MAE en folds de entrenamiento",
        "permutaciones_por_fold": 5,
        "test_usado_para_seleccion": False,
        "atributos": nombres,
        "sha256_particion": hashlib.sha256(
            (DATOS / "particion.npz").read_bytes()
        ).hexdigest(),
    }
    guardar_json(TABLAS / "protocolo.json", protocolo)
    fold_por_fila = np.empty(len(y), dtype=int)
    for numero, (_, validacion) in enumerate(folds):
        fold_por_fila[validacion] = numero
    pd.DataFrame(
        {"indice_en_entrenamiento": np.arange(len(y)), "fold": fold_por_fila}
    ).to_csv(TABLAS / "folds.csv", index=False)
    registros, mejores, modelos = [], {}, {}
    for nombre, grilla in candidatos().items():
        mejor_mae, mejor_param = np.inf, None
        for configuracion in grilla:
            errores = []
            for fold, (train, validacion) in enumerate(folds):
                modelo = construir(nombre, configuracion).fit(x[train], y[train])
                pred = predecir(modelo, nombre, x[validacion])
                mae = float(mean_absolute_error(y[validacion], pred))
                errores.append(mae)
                registros.append(
                    {
                        "modelo": nombre,
                        "parametros": json.dumps(configuracion, sort_keys=True),
                        "fold": fold,
                        "mae": mae,
                        "n_validacion": len(validacion),
                    }
                )
            promedio = float(np.mean(errores))
            print(f"CV {nombre} {configuracion}: MAE={promedio:.6f}", flush=True)
            if promedio < mejor_mae:
                mejor_mae, mejor_param = promedio, configuracion
        if mejor_param is None:
            raise RuntimeError("No se seleccionó configuración.")
        modelos[nombre] = construir(nombre, mejor_param).fit(x, y)
        mejores[nombre] = {"parametros": mejor_param, "mae_cv": mejor_mae}
    pd.DataFrame(registros).to_csv(TABLAS / "cv.csv", index=False)
    mejor_boosting = min(("M4", "M5", "M6"), key=lambda m: mejores[m]["mae_cv"])
    seleccion = {
        "modelos": mejores,
        "mejor_boosting_cv": mejor_boosting,
        "segundos": perf_counter() - inicio,
        "seleccion_solo_entrenamiento": True,
    }
    joblib.dump(modelos, DATOS / "modelos.joblib")
    guardar_json(TABLAS / "seleccion.json", seleccion)
    print(json.dumps(seleccion, indent=2), flush=True)


def evaluar() -> None:
    marcador = DATOS / "test-evaluado.json"
    if marcador.exists():
        print("La evaluación final ya existe; no se vuelve a consultar prueba.")
        return
    if not (TABLAS / "diagnostico-metadatos.json").exists():
        raise RuntimeError(
            "Completar primero el diagnóstico exclusivamente en entrenamiento."
        )
    modelos = joblib.load(DATOS / "modelos.joblib")
    seleccion = json.loads((TABLAS / "seleccion.json").read_text())
    # Congelar huellas antes de acceder a las etiquetas de prueba.
    guardar_json(
        marcador,
        {
            "estado": "iniciada",
            "sha256_modelos": hashlib.sha256(
                (DATOS / "modelos.joblib").read_bytes()
            ).hexdigest(),
            "sha256_seleccion": hashlib.sha256(
                (TABLAS / "seleccion.json").read_bytes()
            ).hexdigest(),
        },
    )
    x, y = cargar_prueba()
    filas = []
    predicciones = {"real": y}
    for nombre, modelo in modelos.items():
        pred = predecir(modelo, nombre, x)
        predicciones[nombre] = pred
        filas.append(
            {
                "modelo": nombre,
                "exactitud": float(accuracy_score(y, pred)),
                "mae": float(mean_absolute_error(y, pred)),
                "qwk": float(
                    cohen_kappa_score(y, pred, labels=NIVELES, weights="quadratic")
                ),
            }
        )
    resultados = pd.DataFrame(filas).sort_values(["mae", "modelo"])
    resultados.to_csv(TABLAS / "prueba-metricas.csv", index=False)
    pd.DataFrame(predicciones).to_csv(DATOS / "prueba-predicciones.csv", index=False)
    ganador = str(resultados.iloc[0]["modelo"])
    matriz = confusion_matrix(y, predicciones[ganador], labels=NIVELES)
    pd.DataFrame(matriz, index=NIVELES, columns=NIVELES).to_csv(
        TABLAS / "confusion.csv"
    )
    q = modelos["M6"].named_steps["modelo"].probabilidades_crudas(x)
    violaciones = float(np.mean(np.any(np.diff(q, axis=1) > 1e-12, axis=1)))
    resumen = {
        "ganador_mae": ganador,
        "ganador_exactitud": str(
            resultados.sort_values(
                ["exactitud", "modelo"], ascending=[False, True]
            ).iloc[0]["modelo"]
        ),
        "ganador_qwk": str(
            resultados.sort_values(["qwk", "modelo"], ascending=[False, True]).iloc[0][
                "modelo"
            ]
        ),
        "fraccion_violaciones_M6_antes_PAVA": violaciones,
        "tolerancia_violaciones": 1e-12,
        "n_prueba": len(y),
        "mejor_boosting_seleccionado_antes_de_prueba": seleccion["mejor_boosting_cv"],
    }
    guardar_json(TABLAS / "evaluacion-final.json", resumen)
    guardar_json(marcador, {**json.loads(marcador.read_text()), "estado": "completa"})
    print(resultados.to_string(index=False), flush=True)
    print(json.dumps(resumen, indent=2), flush=True)
    dibujar_confusion(matriz, ganador)


def dibujar_confusion(matriz: np.ndarray, ganador: str) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(RESULTADOS / ".cache" / "matplotlib"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figura = Figure(figsize=(7, 6), layout="constrained")
    FigureCanvasAgg(figura)
    eje = figura.subplots()
    imagen = eje.imshow(matriz, cmap="Blues")
    figura.colorbar(imagen, ax=eje, label="Número de vinos")
    for i in range(7):
        for j in range(7):
            eje.text(
                j,
                i,
                str(matriz[i, j]),
                ha="center",
                va="center",
                color="white" if matriz[i, j] > matriz.max() / 2 else "black",
            )
    eje.set_xticks(range(7), labels=NIVELES)
    eje.set_yticks(range(7), labels=NIVELES)
    eje.set_xlabel("Calidad predicha")
    eje.set_ylabel("Calidad real")
    eje.set_title(f"Prueba final: {ganador}, elegido por MAE")
    (RESULTADOS / "figuras").mkdir(parents=True, exist_ok=True)
    figura.savefig(RESULTADOS / "figuras" / "confusion.png", dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fase", choices=["seleccionar", "evaluar"])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        seleccionar() if args.fase == "seleccionar" else evaluar()


if __name__ == "__main__":
    main()
