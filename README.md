# Transformer desde cero

Proyecto pequeño, hecho a mano y con propósito didáctico: construir desde cero un *transformer* sencillo para familiarizarse con sus piezas y entender su funcionamiento interno.

En lugar de depender de frameworks de aprendizaje profundo, se ha usado **NumPy** por comodidad para las operaciones con matrices. Esto permite que los cálculos principales —incluida la pasada hacia delante y la actualización de pesos— se puedan seguir directamente en el código.

## Qué incluye

- Tokenización básica de archivos de texto.
- Embeddings aprendibles.
- Atención causal multi-cabeza con matrices Query, Key y Value.
- Red *feed-forward* con activación ReLU.
- Cabeza de modelo de lenguaje para predecir el siguiente token.
- Entrenamiento sencillo mediante retropropagación y generación autorregresiva.

Los textos de entrenamiento de ejemplo se encuentran en `data/`.

## Requisitos

- Python 3.12 o superior.
- NumPy.

Las dependencias están declaradas en `pyproject.toml`.

## Ejecución

Con `uv`:

```bash
uv run src/main.py
```

El script entrena el modelo con cada archivo `.txt` de `data/`

## Estructura

```text
src/
  attention.py       # atención multi-cabeza y retropropagación
  feed_forward.py    # capa feed-forward
  lm_head.py         # logits, softmax y pérdida
  my_transformer.py  # composición, entrenamiento y predicción
  tokenizer.py       # lectura, tokenización y lotes
  main.py            # punto de entrada de ejemplo
data/                # corpus de entrenamiento
```

## Alcance

No busca ser una implementación completa ni optimizada para producción. Es un experimento de aprendizaje pensado para explorar los fundamentos de un transformer de forma clara y práctica.
