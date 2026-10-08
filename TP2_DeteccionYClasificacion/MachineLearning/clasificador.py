import cv2
import numpy as np
from joblib import load

# ---------------------------------------------------------------------------
# Clasificador
#
# Aplicación 3 de 3. Versión del detector de Proyecto 1 que reemplaza
# matchShapes por la predicción de un modelo de machine learning
# (modelo.joblib, generado por entrenador.py) entrenado sobre invariantes
# de Hu.
# ---------------------------------------------------------------------------

ETIQUETAS = {1: "Reloj", 2: "Hilo", 3: "Lentes"}
ARCHIVO_MODELO = "modelo.joblib"
CAMARA_INDICE = 1  # cambiar a 0 si la webcam no aparece en el índice 1

# Umbral de confianza mínima para aceptar una predicción como válida.
# El árbol de decisión expone predict_proba(); si la clase más probable
# no supera este umbral, se muestra como "Desconocido".
UMBRAL_CONFIANZA = 0.5


def nada(x):
    pass


def calcular_hu(contorno):
    momentos = cv2.moments(contorno)
    hu = cv2.HuMoments(momentos).flatten()
    return hu.reshape(1, -1)


def main():
    print(f"Cargando modelo desde {ARCHIVO_MODELO} ...")
    clasificador = load(ARCHIVO_MODELO)

    cap = cv2.VideoCapture(CAMARA_INDICE)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara en el índice {CAMARA_INDICE}.")
        return

    cv2.namedWindow('Controles')
    cv2.resizeWindow('Controles', 400, 150)
    cv2.createTrackbar('Umbral', 'Controles', 127, 255, nada)
    cv2.createTrackbar('Tam_Morfologia', 'Controles', 5, 20, nada)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        val_umbral = cv2.getTrackbarPos('Umbral', 'Controles')
        _, binaria = cv2.threshold(gris, val_umbral, 255, cv2.THRESH_BINARY_INV)

        tam_morf = cv2.getTrackbarPos('Tam_Morfologia', 'Controles')
        if tam_morf > 0:
            kernel = np.ones((tam_morf, tam_morf), np.uint8)
            binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
            binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('Mascara (Paso Intermedio)', binaria)

        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            if cv2.contourArea(cnt) < 500:
                continue

            hu = calcular_hu(cnt)
            etiqueta_predicha = clasificador.predict(hu)[0]

            # Confianza de la predicción, si el modelo la soporta
            confianza = None
            if hasattr(clasificador, "predict_proba"):
                probabilidades = clasificador.predict_proba(hu)[0]
                confianza = max(probabilidades)

            x, y, w, h = cv2.boundingRect(cnt)

            if confianza is None or confianza >= UMBRAL_CONFIANZA:
                nombre = ETIQUETAS.get(etiqueta_predicha, f"Clase {etiqueta_predicha}")
                color = (0, 255, 0)
                if confianza is not None:
                    texto = f"{nombre} ({confianza:.0%})"
                else:
                    texto = nombre
            else:
                color = (0, 0, 255)
                texto = f"Desconocido ({confianza:.0%})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Output - Clasificacion ML', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
