# Comparación de segmentadores de calle: TwinLiteNetPlus vs Ultra-Fast-Lane-Detection v2

Kit para correr los dos modelos sobre los mismos videos (calles de la UNLaM y vista de cámara
a bordo de BFMC) y obtener una comparación cuantitativa y cualitativa lista para el informe.

## Qué hace cada modelo (importante para interpretar)

| | TwinLiteNetPlus | UFLDv2 |
|---|---|---|
| Tipo de salida | **Segmentación** por píxel: área manejable + líneas de carril | **Detección** de hasta 4 carriles como puntos sobre filas/columnas fijas (anclas) |
| Entrenado en | BDD100K (calles reales, EE.UU.) | CULane (China) / TuSimple (autopistas EE.UU.) |
| Tamaño | 0,03 M (nano) a 1,94 M (large) parámetros | ResNet18 + cabeza densa ≈ 206 M parámetros (CULane) |
| Entrada | letterbox a 640 px | 1600×320 (CULane) u 800×320 (TuSimple), parte inferior de la imagen |

Como las salidas son distintas, el script usa **una métrica en común**: en una fila cerca de la parte
de abajo de la imagen (`--fila-eval`, 85 % de la altura por defecto) busca el borde izquierdo y
derecho del carril propio y calcula el **centro del carril**. Eso es justamente lo que necesita un
auto (real o el de BFMC) para mantenerse en el carril, y permite comparar ambos modelos cuadro a cuadro.

## 1. Instalación

Necesitás Python 3.9+ y, en lo posible, una GPU NVIDIA (en CPU también anda, a 2–5 FPS).
Si no tenés GPU, usá el notebook `Comparacion_segmentadores.ipynb` en Google Colab (GPU gratis).

```bash
bash preparar.sh
```
Clona los dos repos en `repos/`, instala dependencias y baja los pesos de Google Drive a `pesos/`.
Si `gdown` falla (Drive a veces limita descargas), bajalos a mano de los links del README
de cada repo y ponelos en `pesos/twinlite/` y `pesos/ufld/`.

## 2. Videos

```
videos/
  unlam/   <- videos grabados en las calles de la UNLaM (cámara al frente, a la altura del parabrisas)
  bfmc/    <- videos desde la cámara del auto miniatura (no cenital)
```
Cada subcarpeta se toma como un "grupo" y se resume por separado.

**Consejos para grabar en la UNLaM:** celular fijo en el parabrisas (soporte), horizontal, mirando
al frente y con el horizonte más o menos a 40 % de la altura de la imagen; 720p o 1080p a 30 fps;
algunos tramos con líneas bien pintadas, otros con líneas gastadas, sombras de árboles, curvas y,
si se puede, de noche o con lluvia. Clips de 30–60 s alcanzan.

**Videos BFMC (vista de la cámara del auto):** opciones, de mejor a peor:
- Si la cátedra o un equipo de la UNLaM tiene el auto: grabar con su cámara (Raspberry Pi Camera).
- El simulador oficial (Gazebo) del repo de la competencia, github.com/ECC-BFMC, publica la imagen
  de la cámara del auto; se puede grabar con `ros2 bag` / `rosbag` y exportar a mp4.
- Videos que publican los equipos en YouTube con la vista de la cámara a bordo (bajalos con `yt-dlp`
  y recortá solo los tramos de cámara a bordo, sin las tomas cenitales).
- Si solo conseguís imágenes sueltas en secuencia: `ffmpeg -framerate 15 -pattern_type glob -i 'img/*.jpg' -c:v libx264 -pix_fmt yuv420p bfmc.mp4`

## 3. Correr la comparación

```bash
python comparar.py --videos videos/unlam videos/bfmc --salida resultados \
    --tl-weight pesos/twinlite/large.pth --tl-config large \
    --ufld-weight pesos/ufld/culane_res18.pth --ufld-config culane_res18
```
(Ajustá la ruta de `large.pth` según cómo quedó la carpeta descargada: `find pesos -name "*.pth"`.)

Opciones útiles:
- `--max-frames 300` o `--stride 2` para pruebas rápidas.
- `--ufld-config tusimple_res18 --ufld-weight pesos/ufld/tusimple_res18.pth` para la otra variante.
- `--tl-config nano|small|medium|large` (con el `.pth` que corresponda) para ver el compromiso velocidad/calidad.
- `--recorte-sup 0.2` descarta el 20 % superior (cielo). Muy útil en BFMC: UFLDv2 asume que la ruta
  empieza en ~42 % de la altura de la imagen (CULane), y la cámara del auto miniatura suele estar más
  baja e inclinada. Probá con y sin recorte y reportalo.
- `--recorte-inf 0.1` si en el video se ve el capó.
- `--half` usa FP16 en GPU para TwinLiteNetPlus (como el demo original).
- `--device cpu` para forzar CPU (para medir cómo andaría en una Raspberry/Jetson sin GPU).

## 4. Qué genera

```
resultados/
  resumen.csv / resumen.md     tabla final (un renglón por video y modelo)
  graficos_resumen.png         FPS, % de cuadros con ambos bordes y jitter, por grupo
  <grupo>/<video>/
     comparacion.mp4           lado a lado: izq. TwinLiteNetPlus, der. UFLDv2
     centro_carril.png         centro de carril estimado por cada modelo en el tiempo
     capturas/*.png            cuadros de muestra para el informe
     frames.csv                todo cuadro por cuadro
```
En el video: TwinLiteNetPlus pinta en **verde el área manejable** y en **rojo las líneas**; UFLDv2
dibuja puntos por carril (**verde y amarillo = bordes del carril propio**, naranja y violeta = carriles
vecinos). La línea blanca horizontal es la fila de medición; la cruz amarilla, el centro del carril.

### Métricas de `resumen.csv`
| Columna | Significado |
|---|---|
| `lat_modelo_ms_mediana`, `_p95` | latencia solo de la red (sin pre/post-proceso) |
| `fps_modelo`, `fps_total` | FPS solo red / incluyendo pre y post-proceso |
| `pct_ambos_bordes` | % de cuadros donde encontró borde izq. **y** der. del carril propio (≈ tasa de detección) |
| `pct_algun_borde` | % de cuadros con al menos un borde |
| `jitter_centro_mediana` | variación típica del centro entre cuadros consecutivos (fracción del ancho). Menor = más estable |
| `pct_saltos_grandes` | % de cuadros donde el centro salta más de 5 % del ancho (detecciones erráticas) |
| `ancho_carril_std` | dispersión del ancho del carril detectado (en una toma estable debería ser chica) |
| `desacuerdo_centro_mediana` | diferencia típica entre el centro de un modelo y el otro |
| `pct_area_manejable_media`, `pct_lineas_media` | solo TwinLiteNetPlus |
| `n_carriles_media` | solo UFLDv2 |

**Limitación a declarar en el informe:** sin etiquetas (ground truth) estas métricas miden
disponibilidad, estabilidad y velocidad, no exactitud. Si quieren exactitud, etiqueten 20–30 cuadros
por grupo (por ejemplo con CVAT o LabelMe, marcando las líneas como polilíneas), y comparen el error
en píxeles de los bordes en la fila de medición con lo que da `frames.csv` (`x_izq`, `x_der`).

## 5. Estructura sugerida del informe

1. **Introducción**: segmentación vs detección de carriles; por qué importa para conducción autónoma y BFMC.
2. **Modelos**: arquitectura de cada uno (TwinLiteNetPlus: encoder liviano + atención + dos decoders;
   UFLDv2: backbone ResNet + clasificación ordinal sobre anclas híbridas fila/columna), datasets de
   entrenamiento, parámetros.
3. **Metodología**: hardware, videos (cuántos, duración, resolución, condiciones), parámetros de
   corrida (`config_corrida.json`), métricas.
4. **Resultados UNLaM**: tabla, gráficos, capturas lado a lado en casos buenos y en fallas
   (sombras, líneas gastadas, cruces, autos estacionados, rotondas).
5. **Resultados BFMC**: idem; discutir el cambio de dominio (escala, cámara baja, pista de juguete),
   efecto de `--recorte-sup`.
6. **Discusión**: velocidad vs robustez; qué entrega cada uno para controlar el auto (UFLDv2 da
   directamente polilíneas de carril; TwinLiteNetPlus además da área manejable, útil en calles sin
   líneas pintadas, frecuentes en el conurbano); cuál cabe en una Raspberry Pi / Jetson.
7. **Conclusiones y trabajo futuro**: fine-tuning con imágenes propias (UNLaM / BFMC).
