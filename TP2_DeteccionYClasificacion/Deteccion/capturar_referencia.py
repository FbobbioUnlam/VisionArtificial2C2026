"""
Ayudante para sacar las tres fotos de referencia con la webcam.

Muestra la camara y, encima, el contorno que vision.contorno_principal() va a
extraer de esa imagen. Asi se ve antes de disparar si la foto va a servir: si
el contorno no sigue el borde del objeto, hay que mejorar el fondo o la luz.

Uso:
    python capturar_referencia.py

Teclas:
    1   guardar como referencias/lentes.jpg
    2   guardar como referencias/Reloj.jpg
    3   guardar como referencias/hilo.jpg
    q   salir

Para que salga bien:
  - un solo objeto en el cuadro,
  - fondo liso y bien contrastado (una hoja blanca o una mesa oscura),
  - el objeto entero adentro y sin tocar los bordes,
  - que ocupe buena parte del cuadro.
"""

import os
import sys

import cv2

import VisionArtificial2C2026.TP2_DeteccionYClasificacion.Deteccion.vision as vision


CARPETA_REFERENCIAS = "referencias"
VENTANA = "Capturar referencia"

ANCHO_CAMARA = 640
ALTO_CAMARA = 480

# El margen que fondo_es_claro() usa como fondo puro: el objeto no lo debe pisar.
MARGEN_GUIA = 20

# Frames que se descartan al abrir, para que la exposicion se estabilice.
FRAMES_DE_CALENTAMIENTO = 15

TECLAS = {
    ord("1"): "lentes",
    ord("2"): "Reloj",
    ord("3"): "hilo",
}


def abrir_camara(indice=0):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        camara = cv2.VideoCapture(indice, backend)
        if camara.isOpened():
            camara.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAMARA)
            camara.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAMARA)

            # La exposicion automatica tarda en estabilizarse. Sin esto las
            # primeras fotos salen subexpuestas, y una referencia oscura hace
            # que despues no se reconozca nada.
            for _ in range(FRAMES_DE_CALENTAMIENTO):
                camara.read()

            return camara
        camara.release()

    return None


def dibujar_previsualizacion(frame, contorno):
    """Frame con el contorno detectado, la guia de margen y las instrucciones."""

    vista = frame.copy()
    alto, ancho = vista.shape[:2]

    cv2.rectangle(vista, (MARGEN_GUIA, MARGEN_GUIA),
                  (ancho - MARGEN_GUIA, alto - MARGEN_GUIA), (90, 90, 90), 1)

    if vision.parece_negra(vista):
        estado = vision.AVISO_CAMARA_NEGRA
        color = (0, 0, 255)
    elif contorno is None:
        estado = "No se detecta ningun objeto"
        color = (0, 0, 255)
    elif vision.toca_el_borde(contorno, vista.shape):
        # Es el error que arruina una referencia sin que se note despues: el
        # objeto cortado por el borde, o una sombra del fondo pegada al margen.
        cv2.drawContours(vista, [contorno], -1, (0, 0, 255), 2)
        estado = "El contorno toca el borde: centra el objeto y alejalo del margen"
        color = (0, 0, 255)
    else:
        cv2.drawContours(vista, [contorno], -1, (0, 255, 0), 2)
        area = cv2.contourArea(contorno)
        porcentaje = 100.0 * area / (alto * ancho)
        estado = f"Contorno detectado: {int(area)} px ({porcentaje:.1f}% del cuadro)"
        color = (0, 255, 0) if porcentaje > 2 else (0, 215, 255)

    lineas = [
        (estado, color),
        ("1 lentes   2 Reloj   3 hilo   q salir", (255, 255, 255)),
    ]
    for i, (texto, tono) in enumerate(lineas):
        y = 24 + i * 22
        cv2.putText(vista, texto, (10, y), vision.FUENTE, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vista, texto, (10, y), vision.FUENTE, 0.55, tono, 1, cv2.LINE_AA)

    return vista


def guardar(frame, nombre):
    os.makedirs(CARPETA_REFERENCIAS, exist_ok=True)
    ruta = os.path.join(CARPETA_REFERENCIAS, nombre + ".jpg")
    cv2.imwrite(ruta, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return ruta


def main():
    indice = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    camara = abrir_camara(indice)
    if camara is None:
        print(f"No se pudo abrir la camara {indice}.")
        return 1

    print("1 lentes | 2 Reloj | 3 hilo | q salir")

    while True:
        ok, frame = camara.read()
        if not ok:
            print("Se corto la lectura de la camara.")
            break

        # Mismo espejo que main.py, para que las referencias y el video en vivo
        # esten siempre en la misma orientacion.
        frame = cv2.flip(frame, 1)

        contorno = vision.contorno_principal(frame)
        cv2.imshow(VENTANA, dibujar_previsualizacion(frame, contorno))

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla in TECLAS:
            if contorno is None:
                print("No hay contorno para guardar. Mejora el fondo o la luz.")
            else:
                print("Guardado en", guardar(frame, TECLAS[tecla]))

        if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
            break

    camara.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
