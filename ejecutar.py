"""Ejecuta la práctica completa o sus comprobaciones desde cualquier directorio."""

import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter

RAIZ = Path(__file__).resolve().parent


def lanzar(argumentos: list[str], carpeta: Path = RAIZ) -> None:
    print(f"\n> {' '.join(argumentos)}", flush=True)
    subprocess.run([sys.executable, "-u", *argumentos], cwd=carpeta, check=True)


def problema_1() -> None:
    for argumentos in (
        ["problema-1/src/cota.py"],
        ["problema-1/src/conteo.py"],
        ["problema-1/src/aprendizaje.py"],
        ["problema-1/src/seno.py", "precision"],
        ["problema-1/src/seno.py", "rejillas", "--sensibilidad"],
    ):
        lanzar(argumentos)


def comprobar() -> None:
    for problema in ("problema-1", "problema-2"):
        lanzar(["-m", "unittest", "discover", "-s", "tests", "-v"], RAIZ / problema)
    lanzar(["-m", "ruff", "check", "."])
    lanzar(["-m", "ruff", "format", "--check", "."])
    lanzar(["-m", "ty", "check"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "accion",
        nargs="?",
        default="todo",
        choices=["todo", "problema-1", "problema-2", "comprobar"],
    )
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 14, 9):
        parser.error(
            "Este experimento requiere Python 3.14.9, fijado en .python-version."
        )
    inicio = perf_counter()
    try:
        if args.accion == "comprobar":
            comprobar()
        else:
            if args.accion in ("todo", "problema-1"):
                problema_1()
            if args.accion in ("todo", "problema-2"):
                lanzar(["problema-2/ejecutar.py", "todo"])
    except subprocess.CalledProcessError as error:
        print(
            f"\nEjecución detenida: un paso terminó con código {error.returncode}.",
            flush=True,
        )
        return error.returncode
    print(
        f"\n{args.accion}: completado en {perf_counter() - inicio:.3f} s.", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
