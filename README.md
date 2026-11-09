# Examen intermedio de Reconocimiento de Patrones

Universidad de Sonora · Semestre 2026-2 · Examen D.

| Dato                 | Valor                                             |
| -------------------- | ------------------------------------------------- |
| Nombre               | Jesús Flores Lacarra                              |
| Expediente y semilla | 221201513                                         |
| a (último dígito)    | 3                                                 |
| b (penúltimo dígito) | 1                                                 |
| Problema 1           | Arcos de la circunferencia unitaria (a mod 4 = 3) |
| Problema 2           | Calidad de vino blanco de UCI (a mod 3 = 0)       |

## Estructura del repositorio

```text
examen-intermedio-rp-2026/
├── README.md                  # README
├── pyproject.toml             # Dependencias y versiones
├── uv.lock                    # Resolución exacta de dependencias
├── .python-version            # Versión de Python
├── .gitignore                 # Gitignore
├── problema-1/                # código, resultados y README.md del problema 1
│   ├── README.md
│   ├── src/
│   ├── tests/
│   └── resultados/
└── problema-2/                # código, resultados y README.md del problema 2
    ├── README.md
    ├── ejecutar.py            # Entrada por fases del experimento
    ├── src/
    ├── tests/
    └── resultados/
```

## Entorno de Python

- **Intérprete:** CPython 3.14.9 (`.python-version`).
- **Requisitos iniciales:** Python 3.14.9 del sistema, uv y conexión para instalar
  dependencias y descargar el archivo original de UCI en el problema 2.
- **Gestor:** uv; entorno local en `.venv/`.
- **Dependencias:** `pyproject.toml`; versiones bloqueadas en `uv.lock`.
- **Gráficas:** Matplotlib (motor Agg), instalado con uv; sin ejecutables gráficos externos.
- **Cómputo por bloques:** NumPy para las rejillas del seno.
- **Desarrollo:** Ruff (lint y formato) y ty (tipos), grupo `dev`.

---

Instalación reproducible

```sh
uv sync --locked
```

Comprobaciones:

```sh
uv run --locked --directory problema-1 python -m unittest discover -s tests -v
uv run --locked --directory problema-2 python -m unittest discover -s tests -v
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

Aplicar formato:

```sh
uv run --locked ruff format .
```

## Reproducir resultados

Desde la raíz, después de `uv sync --locked`:

```sh
uv run --locked python problema-1/src/cota.py
uv run --locked python problema-1/src/conteo.py
uv run --locked python problema-1/src/aprendizaje.py
uv run --locked python problema-1/src/seno.py precision
uv run --locked python problema-1/src/seno.py rejillas --sensibilidad
```

Los parámetros, pruebas rápidas, resultados e interpretación están en
[el README del problema 1](problema-1/README.md), cuya parte práctica está completa.
Para reproducir el problema 2 en un clon limpio:

```sh
uv run --locked python problema-2/ejecutar.py preparar
uv run --locked python problema-2/ejecutar.py verificar
uv run --locked python problema-2/ejecutar.py seleccionar
uv run --locked python problema-2/ejecutar.py diagnostico
uv run --locked python problema-2/ejecutar.py evaluar
uv run --locked python problema-2/ejecutar.py diagnostico-final
uv run --locked python problema-2/ejecutar.py figuras
```

El [README del problema 2](problema-2/README.md) documenta la eliminación de
duplicados, la reserva de prueba, el protocolo MAE, las verificaciones y los
resultados. Los datos y modelos locales no se versionan. Tras evaluar, las fases
de selección quedan bloqueadas en esa carpeta; no se reajusta usando prueba.

La revisión integral se ejecutó en una copia sin entorno, datos ni resultados
previos: los experimentos completos reprodujeron las 17 tablas CSV (salvo tiempos),
el contenido científico de 11 JSON y las 10 figuras originales byte a byte.
Pasaron las 22 pruebas, Ruff, formato y ty. Esta comprobación corresponde al
contenido local; la publicación y los permisos de GitHub son pasos de entrega.

## Entrega y alcance

El informe en PDF se entrega por Teams y la parte práctica mediante el repositorio
de GitHub, que debe ser privado y dar acceso al profesor indicado en Teams.
Los resultados se interpretan dentro de las distribuciones, rejillas y presupuestos
documentados; no se extrapolan automáticamente a otras muestras.

Se excluyen de Git los datos descargados, entornos, cachés, modelos locales,
fuentes LaTeX, PDF y las carpetas locales de redacción. Los README y el código
contienen lo necesario para ejecutar los experimentos sin esas notas.
