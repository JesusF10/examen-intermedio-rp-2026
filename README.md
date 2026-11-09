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
│   └── resultados/
└── problema-2/                # código, resultados y README.md del problema 2
    ├── README.md
    ├── src/
    └── resultados/
```

## Entorno de Python

- **Intérprete:** CPython 3.14.9 (`.python-version`).
- **Gestor:** uv; entorno local en `.venv/`.
- **Dependencias:** `pyproject.toml`; versiones bloqueadas en `uv.lock`.
- **Desarrollo:** Ruff (lint y formato) y ty (tipos), grupo `dev`.

---

Instalación reproducible

```sh
uv sync
```

Comprobaciones:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
```

Aplicar formato:

```sh
uv run --locked ruff format .
```

## Reproducir resultados

Pendiente por definir
