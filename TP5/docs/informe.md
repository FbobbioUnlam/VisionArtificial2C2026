# Informe – TP5: Localización con PnP

**Materia:** Visión Artificial – UNLaM – 2C 2026
**Integrantes:** …
**Fecha:** …

## 1. Objetivo
Estimar en tiempo real la pose (posición y orientación) de una cámara en un sistema de referencia del mundo, a partir de marcadores ArUco de posición conocida, y visualizarla en Foxglove.

## 2. Marco teórico
### 2.1 Modelo de cámara pinhole
- Proyección: `s·[u v 1]ᵀ = K·[R | t]·[X Y Z 1]ᵀ`
- Matriz intrínseca K (fx, fy, cx, cy) y modelo de distorsión radial/tangencial (k1, k2, p1, p2, k3).

### 2.2 Calibración
- Método de Zhang con tablero de ajedrez. Error de reproyección como métrica.

### 2.3 Marcadores ArUco
- Diccionarios, codificación, detección de esquinas y refinamiento subpíxel.

### 2.4 Perspective-n-Point (PnP)
- Problema: dados n puntos 3D y sus proyecciones 2D, y K, hallar R y t.
- Métodos usados: `IPPE_SQUARE` (1 marcador) y `SQPNP` + refinamiento Levenberg–Marquardt (varios marcadores).
- Ambigüedad de la pose planar.

### 2.5 Transformaciones
- `solvePnP` devuelve T_cw. Pose de la cámara en el mundo: `R_wc = Rᵀ`, `t_wc = −Rᵀ·t`.
- Convenciones de ejes OpenCV vs. Foxglove; cuaterniones.

## 3. Implementación
- Arquitectura (ver diagrama del README) y descripción de cada módulo.
- Mapa del mundo utilizado (foto + tabla de posiciones).
- Publicación en Foxglove: tópicos y layout.

## 4. Resultados
### 4.1 Calibración
| Parámetro | Valor |
|---|---|
| Resolución | |
| Imágenes usadas | |
| fx, fy | |
| cx, cy | |
| Distorsión | |
| RMS (px) | |

### 4.2 Precisión de la localización
(Datos de `docs/mediciones.csv`)

| Posición | Real (m) | Estimada (m) | Error (mm) | Desvío (mm) | Reproy. (px) |
|---|---|---|---|---|---|
| 1 | | | | | |

### 4.3 Capturas
- Imagen con ejes del mundo dibujados.
- Panel 3D de Foxglove con cámara y trayectoria.

## 5. Limitaciones y problemas encontrados
- Ambigüedad / "flip" con un solo marcador.
- Sensibilidad a la medida del marcador y a la calibración.
- Desenfoque por movimiento, iluminación, distancia máxima de detección.

## 6. Conclusiones

## 7. Referencias
- OpenCV – Camera Calibration and 3D Reconstruction.
- OpenCV – Detection of ArUco Markers.
- Zhang, Z. (2000). *A flexible new technique for camera calibration.*
- Foxglove SDK – documentación.
- VisiAr – https://github.com/AlejandroSilvestri/VisiAr
