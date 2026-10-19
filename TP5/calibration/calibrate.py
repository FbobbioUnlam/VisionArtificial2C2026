"""Fase 1b - Calibración de la cámara con tablero de ajedrez.

Equivalente al calibrador de VisiAr (github.com/AlejandroSilvestri/VisiAr).
Si usan el de VisiAr, copien su .yml a calibration/camera_params.yml: el
cargador acepta los nombres de nodo habituales (cameraMatrix, distCoeffs, ...).

Uso (desde la carpeta TP5):
    python calibration/calibrate.py --cols 9 --rows 6 --square 0.025

  --square = lado REAL de un cuadrado del tablero en metros (medir con regla).

Criterio de aceptación: error RMS de reproyección < ~0.5 px.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import resolve, save_camera_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9, help="esquinas internas horizontales")
    ap.add_argument("--rows", type=int, default=6, help="esquinas internas verticales")
    ap.add_argument("--square", type=float, required=True, help="lado del cuadrado [m]")
    ap.add_argument("--images", default="calibration/images")
    ap.add_argument("--out", default="calibration/camera_params.yml")
    ap.add_argument("--show", action="store_true", help="mostrar cada detección")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    obj_pts, img_pts, names, size = [], [], [], None

    files = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in resolve(args.images).glob(ext))
    if not files:
        raise SystemExit(f"No hay imágenes en {resolve(args.images)}")

    for f in files:
        img = cv2.imread(str(f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            print(f"  {f.name}: resolución distinta, se descarta")
            continue
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if not found:
            print(f"  {f.name}: tablero NO encontrado")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_pts.append(objp)
        img_pts.append(corners)
        names.append(f.name)
        if args.show:
            cv2.drawChessboardCorners(img, pattern, corners, found)
            cv2.imshow("deteccion", img)
            cv2.waitKey(300)
    if args.show:
        cv2.destroyAllWindows()

    if len(obj_pts) < 10:
        print(f"[ATENCION] solo {len(obj_pts)} imágenes válidas; se recomiendan 20 o más.")
    if not obj_pts:
        raise SystemExit("Ninguna imagen sirvió para calibrar.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)

    # Error por imagen: sirve para detectar y borrar fotos malas
    print("\nError de reproyección por imagen (px):")
    per_image = []
    for name, op, ip, rv, tv in zip(names, obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        # reshape: OpenCV 5 devuelve las esquinas como (N, 2) y projectPoints como (N, 1, 2)
        diff = proj.reshape(-1, 2) - ip.reshape(-1, 2)
        e = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        per_image.append((e, name))
    for e, name in sorted(per_image, reverse=True):
        flag = "  <-- revisar/borrar" if e > 2 * rms else ""
        print(f"  {name}: {e:.3f}{flag}")

    np.set_printoptions(precision=4, suppress=True)
    print(f"\nImágenes usadas: {len(obj_pts)} | resolución: {size[0]}x{size[1]}")
    print(f"RMS global: {rms:.4f} px  ({'OK' if rms < 0.5 else 'ALTO: revisar fotos'})")
    print("K =\n", K)
    print("dist =", dist.ravel())

    save_camera_params(args.out, K, dist, size[0], size[1], rms)
    print(f"\nGuardado en {resolve(args.out)}")


if __name__ == "__main__":
    main()
