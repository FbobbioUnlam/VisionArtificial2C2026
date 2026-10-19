# Hoja de ruta – TP5 Localización con PnP

Cada fase produce algo que se puede probar solo. Marcar con `[x]` a medida que se completa.

## Fase 0 – Setup (½ día)
- [X] Crear el entorno virtual e instalar `requirements.txt`
- [X] Probar el visualizador Open3D con `python tools/fake_pose_publisher.py`
- [X] Correr `python -m tests.test_pose_synthetic` (debe dar OK)
- [ ] Decidir el mapa del mundo (`config/world_map.yml`): 1 marcador (A) o varios (B, recomendada)

## Fase 1 – Calibración (1 día)
- [X] Imprimir el tablero, pegarlo en superficie rígida, medir el cuadrado
- [X ] Capturar 20–30 imágenes (`calibration/capture_images.py`)
- [ X] Calibrar (`calibration/calibrate.py`) → **RMS < 0,5 px**
- [ X] Guardar `calibration/camera_params.yml` en el repo

## Fase 2 – Detección ArUco (½–1 día)
- [X] Generar e imprimir los marcadores al 100 % (`markers/generate_markers.py`)
- [X] Medir el lado real y actualizar `marker_length`
- [X] Pegarlos en la mesa midiendo las posiciones del mapa
- [X] Verificar detección en vivo (`python -m src.detector`)

## Fase 3 – PnP ⭐ (1–2 días)
- [ ] Entender `src/pose_estimator.py`: correspondencias 2D–3D, IPPE_SQUARE vs SQPNP, inversión T_cw → T_wc
- [ ] Verificar los ejes del mundo dibujados sobre la imagen (`drawFrameAxes`)
- [ ] Medir error real con `tools/measure_accuracy.py` en al menos 5 posiciones

## Fase 4 – Visualización Open3D (1–2 días)
- [ ] Correr `python -m src.main` y verificar que la cámara se mueve en la ventana 3D
- [ ] Comprobar que el frustum y los ejes de la cámara coinciden con hacia dónde mira la real
- [ ] Grabar una sesión `--record` (CSV de poses) y graficar x/y/z (opcional)

## Fase 5 – Robustez (1 día)
- [ ] Probar oclusión de marcadores (tapar uno, tapar todos)
- [ ] Ajustar `smoothing_alpha` y `max_reprojection_error`
- [ ] Observar la ambigüedad con 1 solo marcador visto de frente

## Fase 6 – Entrega (1 día)
- [ ] Video demo en pantalla dividida (real + Open3D)
- [ ] Completar `docs/informe.md`
- [ ] README con integrantes

## Reparto sugerido

| Persona | Responsable de |
|---|---|
| 1 | Calibración + marcadores y medición del mapa (Fases 1–2) |
| 2 | PnP, validación y mediciones de error (Fase 3) |
| 3 | Visualización Open3D y grabaciones (Fase 4), arrancando con la pose falsa |
| Todos | Robustez, video e informe (Fases 5–6) |
