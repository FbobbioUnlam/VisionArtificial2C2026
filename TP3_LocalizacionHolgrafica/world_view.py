"""
Composición de la ventana W2D: toma el fondo cenital estático (capturado una
única vez durante el registro) y le superpone, en cada frame, la pose actual
del marcador trackeado (contorno, flecha, coordenadas y ángulo) más los ejes
de referencia.
"""

import config
import draw_utils
import homography


def renderizar(registro, esquinas_marcador):
    """
    Devuelve una copia del fondo cenital de `registro` con el overlay de pose
    dibujado encima, a partir de las esquinas del marcador trackeado en la
    imagen de cámara (esquinas_marcador: array (4,2), en píxeles de imagen).
    """
    canvas = registro.fondo.copy()

    centro_mm, angulo_deg = homography.calcular_pose(esquinas_marcador, registro.H_mm)
    esquinas_px = homography.transformar_puntos(esquinas_marcador, registro.H_vis)
    centro_px = esquinas_px.mean(axis=0)

    centro_canvas_px = (config.LADO_CANVAS_PX / 2.0, config.LADO_CANVAS_PX / 2.0)
    draw_utils.dibujar_ejes(canvas, centro_canvas_px)
    draw_utils.dibujar_pose(canvas, esquinas_px, centro_px, angulo_deg, centro_mm)

    return canvas
