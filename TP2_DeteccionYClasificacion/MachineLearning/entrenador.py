import csv
import random
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# ---------------------------------------------------------------------------
# Entrenador
#
# Aplicación 2 de 3. Lee el dataset generado por generador_descriptores.py
# (dataset.csv: columnas etiqueta, hu1..hu7), entrena un árbol de decisión
# y guarda el modelo resultante en modelo.joblib para que lo use
# clasificador.py.
# ---------------------------------------------------------------------------

ARCHIVO_DATASET = "dataset.csv"
ARCHIVO_MODELO = "modelo.joblib"


def cargar_dataset(ruta):
    X, Y = [], []
    with open(ruta, newline="") as f:
        lector = csv.reader(f)
        encabezado = next(lector)  # descarta encabezado
        for fila in lector:
            if not fila:
                continue
            etiqueta = int(fila[0])
            hu = [float(v) for v in fila[1:8]]
            X.append(hu)
            Y.append(etiqueta)
    return X, Y


def main():
    print(f"Cargando dataset desde {ARCHIVO_DATASET} ...")
    X, Y = cargar_dataset(ARCHIVO_DATASET)
    print(f"Se cargaron {len(X)} muestras.")

    if len(X) < 6:
        print("Advertencia: hay muy pocas muestras para entrenar/evaluar bien.")

    # Mezclar el dataset (por si las filas están agrupadas por etiqueta)
    combinado = list(zip(X, Y))
    random.shuffle(combinado)
    X, Y = zip(*combinado)
    X, Y = list(X), list(Y)

    # Separar en entrenamiento y prueba para tener una idea del desempeño
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y if len(set(Y)) > 1 else None
    )

    clasificador = tree.DecisionTreeClassifier(random_state=42)
    clasificador.fit(X_train, Y_train)

    if X_test:
        predicciones = clasificador.predict(X_test)
        exactitud = accuracy_score(Y_test, predicciones)
        print(f"\nExactitud en el conjunto de prueba: {exactitud:.2%}")
        print(classification_report(Y_test, predicciones, zero_division=0))

    # Reentrenar con TODO el dataset para el modelo final que se va a usar
    clasificador_final = tree.DecisionTreeClassifier(random_state=42)
    clasificador_final.fit(X, Y)

    dump(clasificador_final, ARCHIVO_MODELO)
    print(f"Modelo guardado en: {ARCHIVO_MODELO}")

    # Representación textual del árbol (no requiere matplotlib)
    print("\nÁrbol de decisión entrenado:")
    print(tree.export_text(clasificador_final, feature_names=[f"hu{i+1}" for i in range(7)]))


if __name__ == '__main__':
    main()
