"""Fase 1a - Captura de imágenes del tablero de ajedrez para calibrar.

Uso (desde la carpeta TP5):
    python calibration/capture_images.py --cols 9 --rows 6

  ESPACIO = guardar imagen (solo si se detecta el tablero)
  q       = salir

--cols / --rows = cantidad de esquinas INTERNAS del tablero (no de cuadrados).
Tomar 20-30 fotos variando posición, inclinación y distancia, cubriendo
también las esquinas de la imagen.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_settings, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--out", default="calibration/images")
    args = ap.parse_args()

    cam_cfg = load_settings()["camera"]
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    count = len(list(out.glob("*.png")))

    cap = cv2.VideoCapture(cam_cfg["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    # Desactivar autofoco si la cámara lo permite (cambia la distancia focal)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    pattern = (args.cols, args.rows)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        view = frame.copy()
        cv2.drawChessboardCorners(view, pattern, corners, found)
        cv2.putText(view, f"{frame.shape[1]}x{frame.shape[0]} | guardadas: {count} | "
                    f"{'TABLERO OK' if found else 'sin tablero'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("Captura calibracion (ESPACIO guardar, q salir)", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and found:
            path = out / f"calib_{count:03d}.png"
            cv2.imwrite(str(path), frame)   # se guarda la imagen SIN dibujos
            count += 1
            print("guardada", path)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
