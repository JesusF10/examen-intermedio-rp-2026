"""Integridad, cierre de prueba y descarga en la entrada simplificada."""

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from src import datos, flujo


class FlujoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        self.raiz = Path(self.temporal.name)
        self.datos = self.raiz / "datos"
        self.tablas = self.raiz / "tablas"
        self.datos.mkdir()
        self.tablas.mkdir()
        self.contextos = ExitStack()
        self.contextos.enter_context(patch.object(flujo, "DATOS", self.datos))
        self.contextos.enter_context(patch.object(flujo, "TABLAS", self.tablas))
        self.contextos.enter_context(patch.object(datos, "DATOS", self.datos))

    def tearDown(self) -> None:
        self.contextos.close()
        self.temporal.cleanup()

    def cerrar(self) -> None:
        for archivo in (
            self.datos / "modelos.joblib",
            self.tablas / "seleccion.json",
            self.datos / "particion.npz",
        ):
            archivo.write_bytes(b"contenido congelado")
        huella = hashlib.sha256(b"contenido congelado").hexdigest()
        (self.datos / "test-evaluado.json").write_text(
            json.dumps(
                {
                    "estado": "completa",
                    "sha256_modelos": huella,
                    "sha256_seleccion": huella,
                }
            )
        )
        (self.tablas / "protocolo.json").write_text(
            json.dumps({"sha256_particion": huella})
        )
        (self.tablas / "diagnostico-final-metadatos.json").write_text("{}")

    def test_orden_y_detencion_ante_fallo(self) -> None:
        pasos = []
        with ExitStack() as contextos:
            for modulo, nombre, fase in (
                (flujo.datos, "preparar", "datos"),
                (flujo.ordinal, "main", "gradiente"),
                (flujo.comparacion, "seleccionar", "cv"),
                (flujo.diagnostico, "ejecutar", "diagnostico"),
                (flujo.comparacion, "evaluar", "prueba"),
                (flujo.diagnostico, "descriptivo_final", "final"),
                (flujo.diagnostico, "regenerar_figuras", "figuras"),
            ):
                contextos.enter_context(
                    patch.object(
                        modulo, nombre, side_effect=lambda f=fase: pasos.append(f)
                    )
                )
            flujo.ejecutar()
            self.assertEqual(
                pasos,
                [
                    "datos",
                    "gradiente",
                    "cv",
                    "diagnostico",
                    "prueba",
                    "final",
                    "figuras",
                ],
            )
            pasos.clear()
            with patch.object(
                flujo.ordinal, "main", side_effect=RuntimeError("gradiente incorrecto")
            ):
                with self.assertRaisesRegex(RuntimeError, "gradiente incorrecto"):
                    flujo.ejecutar()
            self.assertEqual(pasos, ["datos"])

    def test_cierre_reutiliza_sin_entrenar_ni_consultar_prueba(self) -> None:
        self.cerrar()
        with (
            patch.object(datos, "preparar", side_effect=AssertionError("no preparar")),
            patch.object(
                datos, "cargar_prueba", side_effect=AssertionError("no prueba")
            ),
            patch.object(
                flujo.comparacion,
                "seleccionar",
                side_effect=AssertionError("no seleccionar"),
            ),
            patch.object(
                flujo.comparacion, "evaluar", side_effect=AssertionError("no evaluar")
            ),
            patch.object(flujo.diagnostico, "descriptivo_final") as final,
            patch.object(flujo.diagnostico, "regenerar_figuras") as figuras,
        ):
            flujo.ejecutar()
            figuras.assert_called_once_with()
            final.assert_not_called()
        self.assertTrue(flujo.verificar_cierre())

    def test_recupera_solo_diagnostico_final_tras_cierre(self) -> None:
        self.cerrar()
        (self.tablas / "diagnostico-final-metadatos.json").unlink()
        with (
            patch.object(flujo.diagnostico, "descriptivo_final") as final,
            patch.object(flujo.diagnostico, "regenerar_figuras") as figuras,
            patch.object(
                flujo.comparacion, "evaluar", side_effect=AssertionError("no evaluar")
            ),
        ):
            flujo.ejecutar()
            final.assert_called_once_with()
            figuras.assert_called_once_with()

    def test_detecta_modelos_seleccion_o_particion_modificados(self) -> None:
        for archivo in (
            self.datos / "modelos.joblib",
            self.tablas / "seleccion.json",
            self.datos / "particion.npz",
        ):
            with self.subTest(archivo=archivo.name):
                self.cerrar()
                archivo.write_bytes(b"modificado")
                with self.assertRaisesRegex(RuntimeError, "huella"):
                    flujo.verificar_cierre()

    def test_cierre_interrumpido_no_reevalua(self) -> None:
        (self.datos / "test-evaluado.json").write_text('{"estado": "iniciada"}')
        with patch.object(flujo.comparacion, "evaluar") as evaluar:
            with self.assertRaisesRegex(RuntimeError, "interrumpida"):
                flujo.ejecutar()
            evaluar.assert_not_called()

    def test_no_sobrescribe_particion_cerrada(self) -> None:
        self.cerrar()
        with patch.object(datos, "inspeccionar") as inspeccionar:
            with self.assertRaisesRegex(RuntimeError, "no se permite sobrescribir"):
                datos.preparar()
            inspeccionar.assert_not_called()
        self.assertTrue(flujo.verificar_cierre())

    def test_rechaza_fuente_distinta_antes_de_escribir_csv(self) -> None:
        with ZipFile(self.datos / "wine-quality.zip", "w") as archivo:
            archivo.writestr("winequality-white.csv", "contenido distinto")
            archivo.writestr("winequality.names", "descripción")
        with self.assertRaisesRegex(RuntimeError, "versión del experimento"):
            datos.descargar()
        self.assertFalse((self.datos / "winequality-white.csv").exists())


if __name__ == "__main__":
    unittest.main()
