"""Descarga explícita de vino blanco, inspección y partición reproducible."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

SEMILLA = 221201513
RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos"
RESULTADOS = RAIZ / "resultados"
URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
SHA256_CSV = "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836"


def descargar() -> Path:
    DATOS.mkdir(parents=True, exist_ok=True)
    archivo = DATOS / "wine-quality.zip"
    if not archivo.exists():
        with urlopen(URL, timeout=120) as respuesta:
            contenido = respuesta.read()
        temporal = archivo.with_suffix(".tmp")
        temporal.write_bytes(contenido)
        temporal.replace(archivo)
    with ZipFile(archivo) as comprimido:
        for nombre in ("winequality-white.csv", "winequality.names"):
            entrada = next(n for n in comprimido.namelist() if n.endswith(nombre))
            contenido = comprimido.read(entrada)
            if (
                nombre.endswith(".csv")
                and hashlib.sha256(contenido).hexdigest() != SHA256_CSV
            ):
                raise RuntimeError(
                    "El CSV de UCI no coincide con la versión del experimento."
                )
            (DATOS / nombre).write_bytes(contenido)
    return DATOS / "winequality-white.csv"


def inspeccionar() -> pd.DataFrame:
    ruta = descargar()
    tabla = pd.read_csv(ruta, sep=";")
    print(f"Filas={len(tabla)}, atributos={len(tabla.columns) - 1}")
    print(tabla["quality"].value_counts().sort_index().to_string())
    print(f"Filas duplicadas exactas={int(tabla.duplicated().sum())}")
    conflictos = tabla.groupby(list(tabla.columns[:-1]))["quality"].nunique()
    print(
        f"Perfiles X con etiquetas diferentes={int(np.count_nonzero(np.asarray(conflictos) > 1))}"
    )
    print(f"Faltantes={int(tabla.isna().sum().sum())}")
    return tabla


def preparar() -> None:
    if (DATOS / "test-evaluado.json").exists():
        raise RuntimeError(
            "Prueba ya cerrada: no se permite sobrescribir su partición."
        )
    original = inspeccionar()
    # Decisión previa a la partición: cada perfil exactamente repetido cuenta una vez.
    sin_repetidos = original.drop_duplicates()
    originales = sin_repetidos.index.to_numpy(dtype=np.int64)
    tabla = sin_repetidos.reset_index(drop=True)
    tabla.to_csv(DATOS / "vino-limpio.csv", index=False)
    indices = np.arange(len(tabla), dtype=np.int64)
    entrenamiento, prueba = train_test_split(
        indices, test_size=0.25, random_state=SEMILLA, stratify=tabla["quality"]
    )
    particion = DATOS / "particion.npz"
    np.savez(particion, entrenamiento=entrenamiento, prueba=prueba)
    tablas = RESULTADOS / "tablas"
    tablas.mkdir(parents=True, exist_ok=True)
    conteos = pd.DataFrame(
        {
            "original": original["quality"].value_counts(),
            "total": tabla["quality"].value_counts(),
            "entrenamiento": tabla.iloc[entrenamiento]["quality"].value_counts(),
            "prueba": tabla.iloc[prueba]["quality"].value_counts(),
        }
    ).sort_index()
    conteos.index.name = "quality"
    conteos.to_csv(tablas / "niveles.csv")
    pd.DataFrame(
        {
            "fila_limpia": indices,
            "fila_original": originales,
            "particion": np.where(
                np.isin(indices, entrenamiento), "entrenamiento", "prueba"
            ),
        }
    ).to_csv(tablas / "particion.csv", index=False)
    metadata = {
        "fuente": URL,
        "semilla": SEMILLA,
        "filas_originales": len(original),
        "filas": len(tabla),
        "atributos": list(tabla.columns[:-1]),
        "niveles": sorted(tabla.quality.unique().tolist()),
        "sha256_zip": hashlib.sha256(
            (DATOS / "wine-quality.zip").read_bytes()
        ).hexdigest(),
        "sha256_csv": hashlib.sha256(
            (DATOS / "winequality-white.csv").read_bytes()
        ).hexdigest(),
        "entrenamiento": len(entrenamiento),
        "prueba": len(prueba),
        "duplicados_exactos": int(original.duplicated().sum()),
        "politica_duplicados": "eliminar filas idénticas antes de dividir; conservar primera aparición",
        "recodificacion": "ninguna: calidad 3..9; índices 0..6 solo internos al ordinal",
    }
    (tablas / "datos-metadatos.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def cargar_entrenamiento() -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    tabla = pd.read_csv(DATOS / "vino-limpio.csv")
    with np.load(DATOS / "particion.npz") as particion:
        indices = particion["entrenamiento"]
    filas = tabla.iloc[indices]
    return (
        filas.iloc[:, :-1].to_numpy(dtype=np.float64),
        filas.quality.to_numpy(dtype=np.int64),
        list(tabla.columns[:-1]),
    )


def cargar_prueba() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Solo invocar después de fijar todos los modelos e hiperparámetros."""
    tabla = pd.read_csv(DATOS / "vino-limpio.csv")
    with np.load(DATOS / "particion.npz") as particion:
        indices = particion["prueba"]
    filas = tabla.iloc[indices]
    return filas.iloc[:, :-1].to_numpy(dtype=np.float64), filas.quality.to_numpy(
        dtype=np.int64
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=["inspeccionar", "preparar"])
    args = parser.parse_args()
    if args.accion == "inspeccionar":
        inspeccionar()
    else:
        preparar()


if __name__ == "__main__":
    main()
