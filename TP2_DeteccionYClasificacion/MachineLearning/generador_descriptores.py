import cv2
import numpy as np
import csv
import os

# ---------------------------------------------------------------------------
# Generador de descriptores
#
# Aplicación 1 de 3. Detecta el contorno de un objeto frente a la webcam y,
# cuando el usuario pulsa la barra espaciadora, calcula sus 7 invariantes de
# Hu y los guarda como una fila en un archivo CSV (dataset.csv), junto con
# la etiqueta numérica del objeto que se está mostrando en ese momento.
#
# Controles:
#   1, 2, 3   -> elegir qué etiqueta se va a grabar (ver diccionario abajo)
#   ESPACIO   -> capturar el contorno actual y guardarlo en el dataset
#   q         -> salir
#
# Sugerencia de uso: para cada objeto, mostrarlo en distintas posiciones,
# rotaciones y distancias de la cámara, presionando ESPACIO varias veces
# en cada una para juntar muestras variadas.
# ---------------------------------------------------------------------------

# Diccionario de etiquetas: número -> nombre del objeto
ETIQUETAS = {1: "Reloj", 2: "Hilo", 3: "Lentes"}

ARCHIVO_DATASET = "dataset.csv"
CAMARA_INDICE = 1  # cambiar a 0 si la webcam no aparece en el índice 1


def nada(x):
    pass


def calcular_hu(contorno):
    momentos = cv2.moments(contorno)
    hu = cv2.HuMoments(momentos).flatten()
    return hu


def guardar_muestra(hu, etiqueta):
    existe = os.path.isfile(ARCHIVO_DATASET)
    with open(ARCHIVO_DATASET, mode="a", newline="") as f:
        escritor = csv.writer(f)
        if not existe:
            encabezado = ["etiqueta"] + [f"hu{i+1}" for i in range(7)]
            escritor.writerow(encabezado)
        escritor.writerow([etiqueta] + list(hu))


def main():
    cap = cv2.VideoCapture(CAMARA_INDICE)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara en el índice {CAMARA_INDICE}.")
        return

    cv2.namedWindow('Controles')
    cv2.resizeWindow('Controles', 400, 150)
    cv2.createTrackbar('Umbral', 'Controles', 127, 255, nada)
    cv2.createTrackbar('Tam_Morfologia', 'Controles', 5, 20, nada)

    etiqueta_actual = 1
    contador_por_etiqueta = {k: 0 for k in ETIQUETAS}

    print("Generador de descriptores listo.")
    print("Teclas: 1/2/3 elegir etiqueta, ESPACIO capturar, q salir.")
    print(f"Etiquetas disponibles: {ETIQUETAS}")

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

        contorno_principal = None
        if contornos:
            contornos_validos = [c for c in contornos if cv2.contourArea(c) > 500]
            if contornos_validos:
                contorno_principal = max(contornos_validos, key=cv2.contourArea)

        if contorno_principal is not None:
            x, y, w, h = cv2.boundingRect(contorno_principal)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(frame, [contorno_principal], -1, (255, 0, 0), 2)

        # Overlay informativo
        nombre_actual = ETIQUETAS.get(etiqueta_actual, "?")
        texto_estado = f"Etiqueta: {etiqueta_actual} ({nombre_actual})  Muestras: {contador_por_etiqueta[etiqueta_actual]}"
        cv2.putText(frame, texto_estado, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "1/2/3: etiqueta | ESPACIO: capturar | q: salir", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Output - Generador de descriptores', frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla in (ord('1'), ord('2'), ord('3')):
            nueva_etiqueta = int(chr(tecla))
            if nueva_etiqueta in ETIQUETAS:
                etiqueta_actual = nueva_etiqueta
        elif tecla == ord(' '):
            if contorno_principal is not None:
                hu = calcular_hu(contorno_principal)
                print(f"[Etiqueta {etiqueta_actual} - {nombre_actual}] Hu: {list(hu)}")
                guardar_muestra(hu, etiqueta_actual)
                contador_por_etiqueta[etiqueta_actual] += 1
            else:
                print("No hay contorno detectado, no se guardó nada.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset guardado en: {os.path.abspath(ARCHIVO_DATASET)}")
    print(f"Totales por etiqueta: {contador_por_etiqueta}")


if __name__ == '__main__':
    main()
