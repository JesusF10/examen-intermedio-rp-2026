# Examen intermedio de Reconocimiento de Patrones

Universidad de Sonora · Semestre 2026-2 · Examen D.

El contenido teórico y práctico y las bitácoras de decisiones fueron revisados
personalmente por el autor.

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
├── ejecutar.py                # Ejecución completa y comprobaciones
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

Instalación explícita del entorno (opcional; `uv run --locked` también lo prepara):

```sh
uv sync --locked
```

Comprobaciones:

```sh
uv run --locked python ejecutar.py comprobar
```

Ejecuta las pruebas de ambos problemas, Ruff, formato y ty. Es una comprobación
rápida; los experimentos completos se ejecutan con el comando siguiente.

Aplicar formato:

```sh
uv run --locked ruff format .
```

## Reproducir resultados

Desde la raíz, un solo comando prepara el entorno bloqueado y ejecuta ambos
problemas con todos los parámetros y presupuestos del examen:

```sh
uv run --locked python ejecutar.py
```

Incluye el conteo exhaustivo, aprendizaje, precisión, rejillas con sensibilidad,
descarga de UCI, verificación de gradientes y statsmodels, CV, evaluación final,
diagnósticos y figuras. Tarda varios minutos, muestra el avance y se detiene
si un paso falla. Las salidas se guardan en `problema-1/resultados/` y
`problema-2/resultados/`. No depende de `docs/` ni del informe LaTeX.

Para ejecutar únicamente un problema:

```sh
uv run --locked python ejecutar.py problema-1
uv run --locked python ejecutar.py problema-2
```

El problema 1 vuelve a calcular sus experimentos. El problema 2 ejecuta todas las
fases en una copia limpia. Si prueba ya está cerrada, verifica las huellas de
modelos, selección y partición, conserva la evaluación y regenera las figuras.
El flujo recupera el diagnóstico final tras una interrupción usando solo entrenamiento.
No elimina marcadores ni vuelve a seleccionar o evaluar con prueba conocida.
Para repetir desde cero el problema 2 se usa otra copia limpia del repositorio.

El CSV de UCI se comprueba contra su SHA-256 fijado para impedir que un cambio
en el origen altere silenciosamente los datos. No se versionan descargas ni modelos.
Los parámetros, fases individuales, pruebas rápidas e interpretación están en
[el README del problema 1](problema-1/README.md) y
[el README del problema 2](problema-2/README.md).

El comando único se verificó desde una copia sin entorno, datos ni resultados
previos: completó ambos experimentos en 393.827 s en el equipo de revisión.
Coincidieron las 17 tablas CSV salvo tiempos, el contenido científico de 11 JSON
y las 10 figuras byte a byte con los resultados originales. Pasaron las 29
pruebas, Ruff, formato y ty. Los tiempos y los últimos bits pueden variar con
la plataforma; las limitaciones numéricas se explican en cada problema.
