"""Entrada pública: imports de paquete estables para guardar y cargar modelos."""

import argparse

from src import comparacion, datos, diagnostico, flujo, ordinal
from threadpoolctl import threadpool_limits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fase",
        choices=[
            "todo",
            "inspeccionar",
            "preparar",
            "verificar",
            "seleccionar",
            "diagnostico",
            "evaluar",
            "diagnostico-final",
            "figuras",
        ],
    )
    args = parser.parse_args()
    acciones = {
        "todo": flujo.ejecutar,
        "inspeccionar": datos.inspeccionar,
        "preparar": datos.preparar,
        "verificar": ordinal.main,
        "seleccionar": comparacion.seleccionar,
        "diagnostico": diagnostico.ejecutar,
        "evaluar": comparacion.evaluar,
        "diagnostico-final": diagnostico.descriptivo_final,
        "figuras": diagnostico.regenerar_figuras,
    }
    with threadpool_limits(limits=1):
        acciones[args.fase]()


if __name__ == "__main__":
    main()
