# TP3 - Localización homográfica

Sistema de localización 2D en tiempo real de un marcador ArUco móvil sobre un
plano, observado en perspectiva por una cámara fija, mediante homografía
plano-a-vista.

## Estado actual: funcional (Fases 1 a 4)

- [x] Fase 1: captura de cámara + detección ArUco + ventana **Cam**
- [x] Fase 2: registro del plano métrico (tecla `r`) → homografías H_mm y H_vis + fondo cenital estático
- [x] Fase 3: overlay en vivo (flecha, ejes, coords/ángulo) sobre la ventana **W2D**
- [x] Fase 4: casos borde (marcador perdido, re-registro, cierre de ventana) + resaltado del marcador trackeado

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

Controles:

- `r` → registrar el plano métrico (todavía no implementado, Fase 2)
- `q` → salir

Si la cámara no abre, cambiá `INDICE_CAMARA` en `config.py` entre `0` y `1`.

## Setup físico

- Marcador ArUco impreso, diccionario `DICT_4X4_50`, 100 mm de lado
  (configurable en `config.py`). Hay uno listo para imprimir en
  `marcador_id0_100mm.png` — imprimilo con escala 100% (sin "ajustar a página")
  para que respete el tamaño real.
- Cámara fija, mirando el escritorio **en ángulo** (no de frente) para que
  se note el efecto de la vista cenital.
- El marcador se mueve a mano sobre ese plano, dentro del cuadro de la cámara.

## Notas para el día de la demo

- La cámara puede no ser el índice `1`: en esta Mac terminó siendo `0`. Si
  `main.py` no abre la cámara, cambiá `INDICE_CAMARA` en `config.py`.
- macOS pide permiso de cámara la primera vez para la app desde la que corras
  el script (Terminal, VS Code, etc.). Se activa en
  **Configuración del Sistema → Privacidad y Seguridad → Cámara**. Si ya
  estaba abierta esa app cuando diste el permiso, cerrala del todo (Cmd+Q) y
  volvé a abrirla.
- Usá `python3`, no `python` (en macOS moderno no viene el alias `python`).
- Para la demo en vivo: (1) mostrar la ventana Cam detectando el marcador,
  (2) apoyar el marcador en el escritorio y apretar `r` para registrar,
  mostrando que aparece la vista cenital, (3) mover el marcador y mostrar
  que W2D lo sigue en tiempo real con coordenadas y ángulo, (4) sacar el
  marcador de cuadro para mostrar que W2D se "congela" en vez de romperse,
  (5) volver a apretar `r` para mostrar el re-registro en caliente.

## Estructura del proyecto

```
TP3/
├── main.py              # loop principal, captura, teclado
├── aruco_detector.py     # detección ArUco (wrapper sobre cv2.aruco)
├── config.py             # parámetros configurables
├── homography.py         # (Fase 2) cálculo de homografías y transformación de puntos
├── world_view.py         # (Fase 2/3) vista cenital estática y su actualización
├── draw_utils.py         # (Fase 3) dibujo de flecha, ejes y texto en W2D
└── requirements.txt
```
