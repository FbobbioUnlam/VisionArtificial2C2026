"""
TP3 - Localización homográfica
Fase 3: localización en tiempo real -> overlay en vivo sobre la ventana W2D.

Controles:
  r -> registrar el plano métrico
  q -> salir
"""

import cv2

import config
import homography
import world_view
from aruco_detector import DetectorAruco


def main():
    detector = DetectorAruco()

    cap = cv2.VideoCapture(config.INDICE_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.ANCHO_FRAME)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ALTO_FRAME)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara (índice {config.INDICE_CAMARA}). "
              f"Probá cambiar INDICE_CAMARA en config.py (0 o 1).")
        return

    registro = None       # homography.Registro, se completa al presionar 'r' con éxito
    id_trackeado = None   # ID del marcador usado para registrar; es el que se sigue después
    ultimo_w2d = None     # último frame de W2D renderizado (se "congela" si se pierde el marcador)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer un frame de la cámara.")
            break

        corners, ids = detector.detectar(frame)
        frame_limpio = frame.copy()  # sin anotaciones, para no "quemarlas" en el fondo cenital
        detector.dibujar_detecciones(frame, corners, ids)
        detector.resaltar_trackeado(frame, corners, ids, id_trackeado)

        n_marcadores = 0 if ids is None else len(ids)
        estado = "REGISTRADO" if registro is not None else "SIN REGISTRAR"
        texto = f"Marcadores: {n_marcadores} | {estado} | r=registrar  q=salir"
        cv2.putText(frame, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Cam", frame)

        if registro is not None:
            # La ventana W2D no se puede actualizar sin marcador detectado: en ese
            # caso se sigue mostrando el último frame válido (se "congela").
            marcador_actual, _ = detector.elegir_marcador(corners, ids, id_objetivo=id_trackeado)
            if marcador_actual is not None:
                ultimo_w2d = world_view.renderizar(registro, marcador_actual[0])
            cv2.imshow("W2D", ultimo_w2d)

        tecla = cv2.waitKey(1) & 0xFF

        # Si se cerró la ventana Cam con el botón de la ventana (la X) en vez de
        # con 'q', cortamos igual para no dejar el proceso corriendo en segundo plano.
        if cv2.getWindowProperty("Cam", cv2.WND_PROP_VISIBLE) < 1:
            break

        if tecla == config.TECLA_SALIR:
            break
        elif tecla == config.TECLA_REGISTRAR:
            marcador_corners, marcador_id = detector.elegir_marcador(corners, ids)
            if marcador_corners is None:
                print("No se detectó ningún marcador: no se puede registrar.")
            else:
                registro = homography.registrar(frame_limpio, marcador_corners[0])
                id_trackeado = marcador_id
                ultimo_w2d = world_view.renderizar(registro, marcador_corners[0])
                print(f"Registro realizado usando el marcador ID {marcador_id}.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
