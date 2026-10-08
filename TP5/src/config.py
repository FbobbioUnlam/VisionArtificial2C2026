"""Carga de configuración, calibración de cámara y mapa del mundo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

# Raíz del proyecto (carpeta TP5). Todas las rutas de los .yml son relativas a ella.
ROOT = Path(__file__).resolve().parent.parent


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_settings(path: str | Path = "config/settings.yml") -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- calibración
@dataclass
class CameraParams:
    K: np.ndarray          # matriz intrínseca 3x3
    dist: np.ndarray       # coeficientes de distorsión (1xN)
    width: int
    height: int


# Nombres posibles de los nodos en el .yml (el nuestro y los habituales de
# otros calibradores, por ej. VisiAr u OpenCV samples).
_K_KEYS = ("camera_matrix", "cameraMatrix", "K", "mtx")
_D_KEYS = ("dist_coeffs", "distCoeffs", "distortion_coefficients", "dist", "D")


def _read_mat(fs: cv2.FileStorage, keys) -> np.ndarray | None:
    for k in keys:
        node = fs.getNode(k)
        if not node.empty():
            return node.mat()
    return None


def load_camera_params(path: str | Path) -> CameraParams:
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No existe {p}. Primero calibrá la cámara: python calibration/calibrate.py"
        )
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
    K = _read_mat(fs, _K_KEYS)
    dist = _read_mat(fs, _D_KEYS)
    w = fs.getNode("image_width")
    h = fs.getNode("image_height")
    width = int(w.real()) if not w.empty() else 0
    height = int(h.real()) if not h.empty() else 0
    fs.release()
    if K is None or dist is None:
        raise ValueError(f"{p} no tiene matriz de cámara o coeficientes de distorsión")
    return CameraParams(K.astype(np.float64), dist.reshape(1, -1).astype(np.float64), width, height)


def save_camera_params(path: str | Path, K, dist, width, height, rms) -> None:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(width))
    fs.write("image_height", int(height))
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.write("rms_reprojection_error", float(rms))
    fs.release()


# ------------------------------------------------------------- mapa del mundo
@dataclass
class WorldMarker:
    id: int
    R: np.ndarray        # rotación marcador -> mundo (3x3)
    t: np.ndarray        # posición del centro del marcador en el mundo (3,)
    corners: np.ndarray  # 4x3, esquinas en el mundo, en el orden de OpenCV


def marker_local_corners(length: float) -> np.ndarray:
    """Esquinas de un marcador en su propio sistema (centro en el origen, z=0).

    Orden de OpenCV: sup-izq, sup-der, inf-der, inf-izq. Es el orden que exige
    SOLVEPNP_IPPE_SQUARE.
    """
    h = length / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def load_world_map(path: str | Path, marker_length: float) -> dict[int, WorldMarker]:
    with open(resolve(path), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    local = marker_local_corners(marker_length)
    world: dict[int, WorldMarker] = {}
    for m in data["markers"]:
        R = rot_z(float(m.get("yaw_deg", 0.0)))
        t = np.asarray(m["position"], dtype=np.float64)
        world[int(m["id"])] = WorldMarker(int(m["id"]), R, t, (R @ local.T).T + t)
    return world
