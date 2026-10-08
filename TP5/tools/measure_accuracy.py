"""Validación (Fase 3/6): compara la posición estimada con una medida real.

Colocar la cámara FIJA en una posición medida con cinta respecto del origen
del mundo, y correr:

    python tools/measure_accuracy.py --real 0.30 0.10 0.45 --frames 100

Promedia N estimaciones y reporta error, desvío estándar (ruido/jitter) y
error de reproyección. Los resultados se agregan a docs/mediciones.csv
para usarlos en el informe.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_camera_params, load_settings, load_world_map, resolve  # noqa: E402
from src.detector import ArucoDetector  # noqa: E402
from src.pose_estimator import PoseEstimator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="posición real de la cámara en el mundo [m]")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cfg = load_settings()
    L = cfg["aruco"]["marker_length"]
    cam = load_camera_params(cfg["camera"]["calibration_file"])
    est = PoseEstimator(cam, load_world_map(cfg["world"]["map_file"], L), L,
                        cfg["pose"]["max_reprojection_error"])
    det = ArucoDetector(cfg["aruco"]["dictionary"])

    cap = cv2.VideoCapture(cfg["camera"]["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])

    positions, errors, tries = [], [], 0
    while len(positions) < args.frames and tries < args.frames * 5:
        ok, frame = cap.read()
        tries += 1
        if not ok:
            break
        pose = est.estimate(det.detect(frame))
        if pose is not None:
            positions.append(pose.t_wc)
            errors.append(pose.reprojection_error)
    cap.release()

    if not positions:
        raise SystemExit("No se obtuvo ninguna pose. ¿Se ven los marcadores?")
    P = np.array(positions)
    mean, std = P.mean(axis=0), P.std(axis=0)
    np.set_printoptions(precision=4, suppress=True)
    print(f"Poses válidas: {len(P)}/{tries}")
    print("Posición estimada media (m):", mean)
    print("Desvío estándar (m):        ", std)
    print(f"Distancia al origen: {np.linalg.norm(mean):.4f} m")
    print(f"Error de reproyección medio: {np.mean(errors):.3f} px")

    row = {"fecha": datetime.now().isoformat(timespec="seconds"), "etiqueta": args.label,
           "est_x": mean[0], "est_y": mean[1], "est_z": mean[2],
           "std_x": std[0], "std_y": std[1], "std_z": std[2],
           "reproj_px": float(np.mean(errors)), "n": len(P)}
    if args.real:
        real = np.array(args.real)
        err = mean - real
        print("Posición real (m):           ", real)
        print("Error (m):                   ", err, f"| norma = {np.linalg.norm(err):.4f} m")
        print(f"Error relativo en distancia: "
              f"{100 * abs(np.linalg.norm(mean) - np.linalg.norm(real)) / np.linalg.norm(real):.2f} %")
        row.update(real_x=real[0], real_y=real[1], real_z=real[2], error_m=float(np.linalg.norm(err)))

    out = resolve("docs/mediciones.csv")
    fields = ["fecha", "etiqueta", "real_x", "real_y", "real_z", "est_x", "est_y", "est_z",
              "error_m", "std_x", "std_y", "std_z", "reproj_px", "n"]
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)
    print("Agregado a", out)


if __name__ == "__main__":
    main()
