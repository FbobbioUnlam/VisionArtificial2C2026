"""
Cálculo de las homografías del TP3 y transformación de puntos entre el plano
de imagen y el sistema de referencia métrico (mm) o de visualización (canvas
cenital de la ventana W2D).

Convención del sistema de referencia del mundo, fijada en el instante del
registro:
  - Origen: centro del marcador usado para registrar
  - Eje x: hacia la derecha, según la orientación impresa del marcador
  - Eje y: hacia arriba, según la orientación impresa del marcador
"""

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class Registro:
    """Resultado de registrar el plano métrico: homografías fijas + fondo cenital."""
    H_mm: np.ndarray     # imagen -> coordenadas en mm
    H_vis: np.ndarray    # imagen -> píxeles del canvas W2D
    fondo: np.ndarray    # vista cenital estática (BGR), capturada una única vez


def _esquinas_destino_mm(lado_mm):
    """Cuadrado ideal en mm, orden TL, TR, BR, BL, centrado en el origen (y hacia arriba)."""
    hs = lado_mm / 2.0
    return np.float32([
        [-hs,  hs],
        [ hs,  hs],
        [ hs, -hs],
        [-hs, -hs],
    ])


def _esquinas_destino_canvas(lado_mm, escala_px_por_mm, lado_canvas_px):
    """
    El mismo cuadrado que _esquinas_destino_mm pero expresado en píxeles de
    canvas: origen en el centro de la imagen, eje y invertido porque en
    píxeles las filas crecen hacia abajo.
    """
    hs_px = (lado_mm / 2.0) * escala_px_por_mm
    centro = lado_canvas_px / 2.0
    return np.float32([
        [centro - hs_px, centro - hs_px],
        [centro + hs_px, centro - hs_px],
        [centro + hs_px, centro + hs_px],
        [centro - hs_px, centro + hs_px],
    ])


def registrar(frame, esquinas_marcador,
              lado_mm=config.LADO_MARCADOR_MM,
              escala_px_por_mm=config.ESCALA_PX_POR_MM,
              lado_canvas_px=config.LADO_CANVAS_PX):
    """
    Registra el plano métrico a partir del marcador detectado en este instante.

    esquinas_marcador: array (4,2) con las esquinas del marcador en la imagen,
    en el orden que devuelve cv2.aruco (TL, TR, BR, BL).

    Devuelve un Registro con ambas homografías (fijas de acá en más) y el
    fondo cenital, generado una única vez con warpPerspective.
    """
    src = np.float32(esquinas_marcador)

    dst_mm = _esquinas_destino_mm(lado_mm)
    dst_vis = _esquinas_destino_canvas(lado_mm, escala_px_por_mm, lado_canvas_px)

    H_mm = cv2.getPerspectiveTransform(src, dst_mm)
    H_vis = cv2.getPerspectiveTransform(src, dst_vis)

    fondo = cv2.warpPerspective(frame, H_vis, (lado_canvas_px, lado_canvas_px))

    return Registro(H_mm=H_mm, H_vis=H_vis, fondo=fondo)


def transformar_puntos(puntos, H):
    """Transforma un array (N,2) de puntos de imagen con la homografía H. Devuelve (N,2)."""
    pts = np.float32(puntos).reshape(-1, 1, 2)
    transformados = cv2.perspectiveTransform(pts, H)
    return transformados.reshape(-1, 2)


def calcular_pose(esquinas_marcador, H_mm):
    """
    Calcula la posición (centro, en mm) y orientación (ángulo, en grados) del
    marcador a partir de sus esquinas en la imagen y la homografía imagen->mm.

    El ángulo se mide con atan2 sobre la dirección TL->TR del marcador ya
    transformado a mm (0° = eje x, sentido antihorario, consistente con y
    hacia arriba).
    """
    esquinas_mm = transformar_puntos(esquinas_marcador, H_mm)
    centro = esquinas_mm.mean(axis=0)

    tl, tr = esquinas_mm[0], esquinas_mm[1]
    direccion = tr - tl
    angulo = float(np.degrees(np.arctan2(direccion[1], direccion[0])))

    return centro, angulo
