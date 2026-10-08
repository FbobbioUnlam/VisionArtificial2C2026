"""
Configuración del sistema de localización homográfica (TP3).
Ajustá estos valores según tu setup físico (cámara, marcador impreso).
"""

import cv2

# -----------------------------
# Cámara
# -----------------------------
INDICE_CAMARA = 0          # tu Mac solo tiene la cámara 0 disponible
ANCHO_FRAME = 1280
ALTO_FRAME = 720

# -----------------------------
# Marcador ArUco
# -----------------------------
DICCIONARIO_ARUCO = cv2.aruco.DICT_4X4_50
LADO_MARCADOR_MM = 100.0   # lado real del marcador impreso, en milímetros

# -----------------------------
# Vista cenital (ventana W2D)
# -----------------------------
ESCALA_PX_POR_MM = 4.0     # escala del canvas cenital (se usa desde la Fase 2)
LADO_CANVAS_PX = 800       # tamaño (cuadrado) de la ventana W2D en píxeles

# -----------------------------
# Teclas
# -----------------------------
TECLA_REGISTRAR = ord('r')
TECLA_SALIR = ord('q')
