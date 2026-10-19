"""Muestra una pose FALSA (cámara girando en círculo) en el visualizador Open3D.

Sirve para armar y probar la visualización (Fase 4) en paralelo, sin tener
la calibración ni la detección listas.

Uso (desde la carpeta TP5):
    python tools/fake_pose_publisher.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_settings, load_world_map  # noqa: E402
from src.pose_estimator import CameraPose  # noqa: E402
from src.visualizer import Open3DVisualizer  # noqa: E402


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """R_wc de una cámara (convención OpenCV: z adelante, y abajo) que mira a target."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(x) < 1e-6:          # mirando justo hacia abajo/arriba
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def main():
    cfg = load_settings()
    L = cfg["aruco"]["marker_length"]
    world = load_world_map(cfg["world"]["map_file"], L)
    center = np.mean([m.t for m in world.values()], axis=0)
    viz = Open3DVisualizer(world, L, trajectory_length=cfg["pose"]["trajectory_length"])
    print("Mostrando pose falsa. Cerrar la ventana o Ctrl+C para salir.")
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            eye = center + np.array([0.5 * np.cos(0.5 * t), 0.5 * np.sin(0.5 * t), 0.4])
            viz.update_pose(CameraPose(look_at(eye, center), eye, 0.0, []))
            if not viz.render():
                break
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()


if __name__ == "__main__":
    main()
