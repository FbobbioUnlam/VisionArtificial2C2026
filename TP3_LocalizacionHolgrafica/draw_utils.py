"""
Utilidades de dibujo para la ventana W2D: ejes de referencia, contorno del
marcador, flecha de pose y texto de coordenadas/ángulo.
"""

import cv2
import numpy as np


def dibujar_ejes(canvas, centro_px, largo_px=60, color=(200, 200, 200), grosor=1):
    """
    Dibuja los ejes canónicos centrados en centro_px: x hacia la derecha,
    y hacia arriba (en píxeles de pantalla, "arriba" es fila decreciente).
    """
    cx, cy = int(centro_px[0]), int(centro_px[1])

    cv2.arrowedLine(canvas, (cx, cy), (cx + largo_px, cy), color, grosor, tipLength=0.15)
    cv2.putText(canvas, "x", (cx + largo_px + 5, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.arrowedLine(canvas, (cx, cy), (cx, cy - largo_px), color, grosor, tipLength=0.15)
    cv2.putText(canvas, "y", (cx + 5, cy - largo_px - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def dibujar_pose(canvas, esquinas_px, centro_px, angulo_deg, centro_mm,
                  color=(0, 200, 255), largo_flecha=50):
    """
    Dibuja sobre canvas (in-place):
      - el contorno cuadrado del marcador (esquinas_px: array (4,2), en píxeles de canvas)
      - una flecha desde el centro indicando la orientación (angulo_deg, sistema mm)
      - texto con las coordenadas en mm y el ángulo en grados
    """
    pts = esquinas_px.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

    cx, cy = centro_px
    angulo_rad = np.radians(angulo_deg)
    # angulo_deg está en el sistema mm (y hacia arriba); en píxeles la fila crece
    # hacia abajo, por eso el seno se resta en vez de sumarse.
    punta = (int(round(cx + largo_flecha * np.cos(angulo_rad))),
             int(round(cy - largo_flecha * np.sin(angulo_rad))))
    cv2.arrowedLine(canvas, (int(round(cx)), int(round(cy))), punta, color, 2, tipLength=0.3)

    texto = f"({centro_mm[0]:.1f}, {centro_mm[1]:.1f}) mm  {angulo_deg:.1f} deg"
    cv2.putText(canvas, texto, (int(round(cx)) + 10, int(round(cy)) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
