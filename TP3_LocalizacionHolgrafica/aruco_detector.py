"""
Detección de marcadores ArUco: wrapper sobre cv2.aruco para detectar
marcadores en un frame y anotarlos (contorno + etiqueta) en la ventana Cam.
"""

import cv2
import numpy as np

import config


class DetectorAruco:

    def __init__(self, diccionario=config.DICCIONARIO_ARUCO):
        self.diccionario = cv2.aruco.getPredefinedDictionary(diccionario)
        self.parametros = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.diccionario, self.parametros)

    def detectar(self, frame):
        """
        Detecta marcadores en el frame (BGR).
        Devuelve (corners, ids):
          - corners: lista de arrays (1,4,2) float32, uno por marcador detectado,
            con las esquinas en orden (sup-izq, sup-der, inf-der, inf-izq)
          - ids: array (N,1) con el ID de cada marcador, o None si no se detectó nada
        """
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rechazados = self.detector.detectMarkers(gris)
        return corners, ids

    def dibujar_detecciones(self, frame, corners, ids):
        """Dibuja el contorno y la etiqueta de cada marcador detectado sobre frame (in-place)."""
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        return frame

    def resaltar_trackeado(self, frame, corners, ids, id_objetivo,
                            color=(255, 0, 255), grosor=3):
        """
        Si el marcador con id_objetivo está entre los detectados, remarca su
        contorno con otro color (in-place) para distinguirlo visualmente de
        otros marcadores que puedan estar en cuadro.
        """
        if id_objetivo is None:
            return frame

        marcador, _ = self.elegir_marcador(corners, ids, id_objetivo=id_objetivo)
        if marcador is None:
            return frame

        pts = marcador.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=grosor)
        return frame

    def elegir_marcador(self, corners, ids, id_objetivo=None):
        """
        Elige qué marcador trackear entre los detectados.
        Si id_objetivo se especifica, busca ese ID puntual; si no está presente, no matchea.
        Si no se especifica, devuelve el primero detectado (criterio simple, válido según el enunciado).

        Devuelve (corners_del_marcador, id) o (None, None) si no hay match.
        """
        if ids is None or len(ids) == 0:
            return None, None

        # ids puede venir con forma (N,1) o (N,), según versión/plataforma de OpenCV;
        # lo aplanamos para no depender de eso.
        ids_planos = np.asarray(ids).reshape(-1)

        if id_objetivo is not None:
            for c, i in zip(corners, ids_planos):
                if int(i) == id_objetivo:
                    return c, int(i)
            return None, None

        return corners[0], int(ids_planos[0])
