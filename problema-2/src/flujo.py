"""Flujo completo y repetición segura después del cierre de prueba."""

import hashlib
import json
from time import perf_counter

from . import comparacion, datos, diagnostico, ordinal
from .comparacion import TABLAS
from .datos import DATOS


def verificar_cierre() -> bool:
    """Verifica huellas de modelos, selección y partición sin acceder a prueba."""
    marcador = DATOS / "test-evaluado.json"
    if not marcador.exists():
        return False
    cierre = json.loads(marcador.read_text(encoding="utf-8"))
    if cierre.get("estado") != "completa":
        raise RuntimeError(
            "La evaluación de prueba quedó interrumpida. Conserve esta carpeta; "
            "para reproducir desde cero use una copia limpia del proyecto."
        )
    for archivo, esperado in (
        (DATOS / "modelos.joblib", cierre["sha256_modelos"]),
        (TABLAS / "seleccion.json", cierre["sha256_seleccion"]),
        (
            DATOS / "particion.npz",
            json.loads((TABLAS / "protocolo.json").read_text(encoding="utf-8"))[
                "sha256_particion"
            ],
        ),
    ):
        if hashlib.sha256(archivo.read_bytes()).hexdigest() != esperado:
            raise RuntimeError(
                f"La huella de {archivo.name} cambió después del cierre."
            )
    return True


def ejecutar() -> None:
    acciones = {
        "preparar": datos.preparar,
        "verificar": ordinal.main,
        "seleccionar": comparacion.seleccionar,
        "diagnostico": diagnostico.ejecutar,
        "evaluar": comparacion.evaluar,
        "diagnostico-final": diagnostico.descriptivo_final,
        "figuras": diagnostico.regenerar_figuras,
    }
    if verificar_cierre():
        print(
            "Prueba ya cerrada; modelos, selección y partición íntegros. "
            "Se reutilizan los resultados sin ajustar ni evaluar prueba.",
            flush=True,
        )
        fases = ["figuras"]
        if not (TABLAS / "diagnostico-final-metadatos.json").exists():
            fases.insert(0, "diagnostico-final")
    else:
        fases = list(acciones)
    for fase in fases:
        print(f"\nProblema 2: {fase}", flush=True)
        inicio = perf_counter()
        acciones[fase]()
        print(f"{fase}: {perf_counter() - inicio:.3f} s", flush=True)
