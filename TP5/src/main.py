"""Programa principal: cámara -> ArUco -> PnP -> visualizador 3D (Open3D).

Uso (desde la carpeta TP5):
    python -m src.main                     # en vivo
    python -m src.main --record            # además graba las poses en data/recordings/*.csv
    python -m src.main --video demo.mp4    # usar un video en lugar de la webcam
    python -m src.main --no-window         # sin ventana de OpenCV

Teclas en la ventana de OpenCV:  q = salir,  r = borrar trayectoria
Cerrar la ventana 3D también termina el programa.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import cv2
import numpy as np

from src.config import load_camera_params, load_settings, load_world_map, resolve
from src.detector import ArucoDetector
from src.filters import PoseSmoother
from src.pose_estimator import PoseEstimator
from src.visualizer import Open3DVisualizer


def overlay(frame, pose, lost_for: float, fps: float) -> None:
    color = (0, 220, 0) if pose is not None and lost_for == 0 else (0, 0, 255)
    if pose is not None:
        x, y, z = pose.t_wc
        lines = [f"cam (m): x={x:+.3f} y={y:+.3f} z={z:+.3f}",
                 f"reproj: {pose.reprojection_error:.2f}px  marcadores: {pose.markers_used}"]
    else:
        lines = ["sin pose"]
    if lost_for > 0:
        lines.append(f"PERDIDA hace {lost_for:.1f}s (se mantiene la ultima)")
    lines.append(f"{fps:.0f} fps")
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description="TP5 - Localización con PnP")
    ap.add_argument("--config", default="config/settings.yml")
    ap.add_argument("--video", help="archivo de video en lugar de la webcam")
    ap.add_argument("--record", action="store_true", help="grabar las poses en CSV")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    cfg = load_settings(args.config)
    cam_cfg, aruco_cfg, pose_cfg = cfg["camera"], cfg["aruco"], cfg["pose"]
    L = float(aruco_cfg["marker_length"])

    cam = load_camera_params(cam_cfg["calibration_file"])
    world = load_world_map(cfg["world"]["map_file"], L)

    cap = cv2.VideoCapture(args.video if args.video else cam_cfg["index"])
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("No se pudo leer la cámara/video")
    h, w = frame.shape[:2]
    if cam.width and (w, h) != (cam.width, cam.height):
        print(f"[ATENCION] la imagen es {w}x{h} pero se calibró a {cam.width}x{cam.height}. "
              "Los intrínsecos no van a ser válidos: recalibrá o cambiá la resolución.")

    detector = ArucoDetector(aruco_cfg["dictionary"])
    estimator = PoseEstimator(cam, world, L, pose_cfg["max_reprojection_error"])
    smoother = PoseSmoother(pose_cfg["smoothing_alpha"])

    record = None
    if args.record:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record = str(resolve(f"data/recordings/sesion_{stamp}.csv"))
    viz = Open3DVisualizer(world, L, cam, (w, h), pose_cfg["trajectory_length"], record)

    last_pose, last_seen = None, 0.0
    t_prev, fps = time.time(), 0.0
    try:
        while ok:
            now = time.time()
            detections = detector.detect(frame)
            pose = estimator.estimate(detections)
            if pose is not None:
                last_pose, last_seen = smoother.update(pose), now
                viz.update_pose(last_pose)
            lost_for = 0.0 if pose is not None else (now - last_seen if last_pose else 0.0)

            detector.draw(frame, detections)
            if pose is not None:
                estimator.draw_axes(frame, pose, length=L)
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            overlay(frame, last_pose, lost_for, fps)

            if not viz.render():          # se cerró la ventana 3D
                break

            if not args.no_window:
                cv2.imshow("TP5 - Localizacion con PnP (q: salir, r: reset trayectoria)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    viz.clear_trajectory()
            ok, frame = cap.read()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        viz.close()
        if last_pose is not None:
            np.set_printoptions(precision=3, suppress=True)
            print("Última posición de la cámara (m):", last_pose.t_wc)


if __name__ == "__main__":
    main()
