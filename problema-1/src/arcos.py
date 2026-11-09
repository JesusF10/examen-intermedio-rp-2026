"""Realizabilidad exacta de un arco, con puntos representados por sus ángulos.

Cada ángulo debe estar en [0, 2*pi). Los extremos del arco son cerrados;
se permiten arcos que cruzan cero, el vacío y la circunferencia completa.
El ERM agrupa puntos repetidos y maximiza una suma circular de pesos enteros.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from math import isfinite, tau
from typing import Literal


def _validar_puntos(puntos: Sequence[float]) -> None:
    if any(not isfinite(angulo) or not 0 <= angulo < tau for angulo in puntos):
        raise ValueError("Los ángulos deben ser finitos y pertenecer a [0, 2*pi).")


def _validar_muestra(puntos: Sequence[float], etiquetas: Sequence[int]) -> None:
    if len(puntos) != len(etiquetas):
        raise ValueError("Debe haber una etiqueta por punto.")
    _validar_puntos(puntos)
    if any(etiqueta not in (-1, 1) for etiqueta in etiquetas):
        raise ValueError("Las etiquetas deben valer -1 o +1.")


@dataclass(frozen=True)
class Arco:
    """Arco cerrado antihorario; tipo explícito para vacío y completo."""

    inicio: float = 0.0
    fin: float = 0.0
    tipo: Literal["arco", "vacio", "completo"] = "arco"

    def __post_init__(self) -> None:
        _validar_puntos((self.inicio, self.fin))
        if self.tipo not in ("arco", "vacio", "completo"):
            raise ValueError("Tipo de arco inválido.")

    def contiene(self, angulo: float) -> bool:
        """Pertenencia de un ángulo ya validado en la representación canónica."""
        if self.tipo == "vacio":
            return False
        if self.tipo == "completo":
            return True
        if self.inicio <= self.fin:
            return self.inicio <= angulo <= self.fin
        return angulo >= self.inicio or angulo <= self.fin

    def predecir(self, puntos: Sequence[float]) -> list[int]:
        _validar_puntos(puntos)
        return [1 if self.contiene(p) else -1 for p in puntos]


def ajustar_erm(puntos: Sequence[float], etiquetas: Sequence[int]) -> Arco:
    """ERM exacto O(n log n), incluidos puntos repetidos con etiquetas en conflicto.

    Cada punto distinto recibe peso igual a la suma de sus etiquetas. Cambiar
    de negativo a positivo ese punto reduce los errores en dicho peso. El óptimo
    es un bloque circular de peso máximo, encontrado con prefijos y una deque.
    Empates: menos puntos distintos positivos y luego menor índice inicial.
    Los extremos son los puntos seleccionados; no se redondean puntos medios.
    """
    _validar_muestra(puntos, etiquetas)
    angulos: list[float] = []
    pesos: list[int] = []
    for angulo, etiqueta in sorted(zip(puntos, etiquetas, strict=True)):
        if angulos and angulo == angulos[-1]:
            pesos[-1] += etiqueta
        else:
            angulos.append(angulo)
            pesos.append(etiqueta)
    m = len(angulos)
    if not m:
        return Arco(tipo="vacio")

    prefijos = list(accumulate(pesos + pesos, initial=0))
    candidatos: deque[int] = deque()
    mejor = (0, 0, 0)  # (ganancia, -longitud, -inicio); vacío gana empates a cero.
    for final in range(1, 2 * m):
        nuevo = final - 1
        if nuevo < m:
            while candidatos and prefijos[candidatos[-1]] >= prefijos[nuevo]:
                candidatos.pop()
            candidatos.append(nuevo)
        while candidatos and candidatos[0] < final - m:
            candidatos.popleft()
        inicio = candidatos[0]
        clave = (prefijos[final] - prefijos[inicio], -(final - inicio), -inicio)
        mejor = max(mejor, clave)

    _, longitud_negativa, inicio_negativo = mejor
    longitud, inicio = -longitud_negativa, -inicio_negativo
    if longitud == 0:
        return Arco(tipo="vacio")
    if longitud == m:
        return Arco(tipo="completo")
    return Arco(angulos[inicio], angulos[(inicio + longitud - 1) % m])


def realizable(puntos: Sequence[float], etiquetas: Sequence[int]) -> bool:
    """Indica si un arco puede reproducir todas las etiquetas dadas.

    Los puntos repetidos se comparan exactamente en la representación angular
    canónica; etiquetas opuestas sobre el mismo punto son imposibles. No se
    fusionan puntos cercanos mediante tolerancias. Costo O(n log n), memoria O(n).
    """
    _validar_muestra(puntos, etiquetas)

    ordenadas: list[int] = []
    anterior: float | None = None
    for angulo, etiqueta in sorted(zip(puntos, etiquetas, strict=True)):
        if angulo == anterior:
            if etiqueta != ordenadas[-1]:
                return False
        else:
            ordenadas.append(etiqueta)
            anterior = angulo

    # Un único bloque circular tiene dos cambios; las constantes tienen cero.
    siguientes = ordenadas[1:] + ordenadas[:1]
    cambios = sum(a != b for a, b in zip(ordenadas, siguientes, strict=True))
    return cambios <= 2
