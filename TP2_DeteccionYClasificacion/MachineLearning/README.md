# Proyecto 2 — Clasificación por invariantes de Hu + Machine Learning

Tres aplicaciones independientes, para usar en este orden:

## 1. Instalar dependencias

```
pip install opencv-python numpy scikit-learn joblib
```

## 2. `generador_descriptores.py`

Abre la webcam y detecta el contorno más grande frente a la cámara.

- Pulsá **1**, **2** o **3** para elegir qué objeto estás mostrando
  (1 = Reloj, 2 = Hilo, 3 = Lentes — editá el diccionario `ETIQUETAS` si
  usás otros objetos).
- Pulsá **ESPACIO** para capturar una muestra: calcula los 7 invariantes
  de Hu del contorno actual y los agrega como fila a `dataset.csv`
  (se crea automáticamente, con encabezado).
- Pulsá **q** para salir.

Mostrá cada objeto en varias posiciones, rotaciones y distancias, capturando
varias muestras de cada uno (cuantas más y más variadas, mejor entrena
el modelo).

## 3. `entrenador.py`

Lee `dataset.csv`, separa una parte para prueba, entrena un
`DecisionTreeClassifier` de scikit-learn, imprime la exactitud obtenida
y el árbol resultante, y guarda el modelo final (entrenado con todo el
dataset) en `modelo.joblib`.

```
python entrenador.py
```

## 4. `clasificador.py`

Igual que el proyecto anterior, pero en lugar de `matchShapes` carga
`modelo.joblib` y usa `predict()` (y `predict_proba()` para mostrar un
porcentaje de confianza) sobre los invariantes de Hu de cada contorno
detectado en la webcam en tiempo real.

```
python clasificador.py
```

## Notas

- Los tres scripts asumen que `dataset.csv` y `modelo.joblib` quedan en el
  mismo directorio desde el que se ejecutan.
- Si la webcam no abre, cambiá `CAMARA_INDICE` de `1` a `0` en
  `generador_descriptores.py` y `clasificador.py`.
- Si el dataset crece mucho o querés otro algoritmo (por ejemplo
  `RandomForestClassifier`), sólo hay que tocar `entrenador.py`; el
  clasificador no necesita cambios porque solo llama a `.predict()`.
