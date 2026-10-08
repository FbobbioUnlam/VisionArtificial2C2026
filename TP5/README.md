# TP5 – Localización con PnP

**Visión Artificial – Ingeniería en Informática – UNLaM – 2C 2026**

> **Consigna:** a partir de la detección de marcadores ArUco, estimar la pose de la cámara con PnP y mostrar su localización en el mundo con un visualizador (Foxglove). Si la cámara se mueve, la pose debe actualizarse en el visualizador. Requiere calibrar la cámara previamente.

## Pipeline

```
Webcam ─► Detección ArUco ─► PnP (2D↔3D) ─► Inversión T_cw → T_wc ─► Filtro ─► Foxglove
              │                    ▲
              │        config/world_map.yml (poses 3D de los marcadores)
              │        calibration/camera_params.yml (K, distorsión)
```

## Estructura

```
TP5/
├── config/
│   ├── settings.yml          # cámara, diccionario, tamaño de marcador, Foxglove
│   └── world_map.yml         # dónde está cada marcador en el mundo (metros)
├── calibration/
│   ├── capture_images.py     # Fase 1: captura fotos del tablero
│   ├── calibrate.py          # Fase 1: calcula K y distorsión -> camera_params.yml
│   └── images/               # fotos del tablero
├── markers/
│   ├── generate_markers.py   # Fase 2: genera los ArUco para imprimir (A4, 300 dpi, PDF)
│   └── output/
├── src/
│   ├── config.py             # carga de yml, calibración y mapa del mundo
│   ├── detector.py           # Fase 2: detección ArUco (también se puede correr solo)
│   ├── pose_estimator.py     # Fase 3: PnP + inversión de la transformación
│   ├── filters.py            # Fase 5: suavizado (EMA + slerp)
│   ├── visualizer.py         # Fase 4: publicación en Foxglove
│   └── main.py               # programa principal
├── tools/
│   ├── fake_pose_publisher.py  # pose falsa para probar Foxglove sin cámara
│   └── measure_accuracy.py     # validación: posición real vs. estimada
├── tests/
│   └── test_pose_synthetic.py  # test de punta a punta con imágenes sintéticas
├── data/recordings/          # grabaciones .mcap
└── docs/
    ├── hoja_de_ruta.md       # plan de trabajo y reparto
    └── informe.md            # plantilla del informe
```

## Instalación

```bash
cd TP5
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Instalar también **Foxglove** (app de escritorio o https://app.foxglove.dev).

> Todos los comandos se corren **desde la carpeta `TP5`**.

## Paso a paso

### 1. Calibrar la cámara
1. Imprimir un tablero de ajedrez (por ej. 10×7 cuadrados → **9×6 esquinas internas**), pegarlo sobre algo rígido y medir el lado del cuadrado.
2. Capturar 20–30 fotos variadas:
   ```bash
   python calibration/capture_images.py --cols 9 --rows 6
   ```
3. Calibrar (ejemplo con cuadrados de 2,5 cm):
   ```bash
   python calibration/calibrate.py --cols 9 --rows 6 --square 0.025
   ```
   Objetivo: **RMS < 0,5 px**. El script marca las fotos con error alto para borrarlas.

*Alternativa:* usar el calibrador de [VisiAr](https://github.com/AlejandroSilvestri/VisiAr/tree/main/Calibration) y copiar su `.yml` a `calibration/camera_params.yml`.

⚠️ Calibrar a la **misma resolución** de `settings.yml` y, si se puede, con el **autofoco desactivado**.

### 2. Marcadores
1. Definir el mapa en `config/world_map.yml` (por defecto: 4 marcadores en un rectángulo de 30×20 cm).
2. Generar e imprimir **al 100 %**:
   ```bash
   python markers/generate_markers.py
   ```
3. **Medir** el lado impreso y corregir `aruco.marker_length` en `settings.yml` si no da exacto.
4. Pegar los marcadores en la mesa según el mapa (la flecha de la hoja apunta a +Y).
5. Probar la detección: `python -m src.detector`

### 3. Correr la localización
```bash
python -m src.main             # en vivo
python -m src.main --record    # además graba data/recordings/sesion_*.mcap
```
En Foxglove: **Open connection → Foxglove WebSocket → `ws://localhost:8765`**.

En el panel **3D**: *Display frame* = `world`. Se ven los marcadores, la cámara (con su frustum gracias a `/camera/calibration`) y la trayectoria. Agregar un panel **Image** con `/camera/image`.

### Herramientas
```bash
python tools/fake_pose_publisher.py                      # probar Foxglove sin cámara
python tools/measure_accuracy.py --real 0.30 0.10 0.45   # medir error (va a docs/mediciones.csv)
python -m tests.test_pose_synthetic                      # verificar PnP con imágenes sintéticas
```

## Convenciones

| Sistema | Ejes |
|---|---|
| Mundo (`world`) | X, Y sobre la mesa, **Z hacia arriba** (igual que Foxglove/ROS) |
| Cámara (`camera`, OpenCV) | X derecha, Y abajo, **Z hacia adelante** |

`solvePnP` devuelve **T_cw** (el mundo visto desde la cámara). La posición de la cámara en el mundo es `t_wc = -Rᵀ·t` y su orientación es `R_wc = Rᵀ`.

## Integrantes
- …
