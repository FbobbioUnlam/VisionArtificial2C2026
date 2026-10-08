"""Fase 3 - Estimación de la pose de la cámara con PnP.

Notación:  T_ab = pose del sistema b expresada en el sistema a.
           Un punto X_b se lleva a 'a' con  X_a = R_ab @ X_b + t_ab.

solvePnP devuelve (rvec, tvec) tales que  X_cam = R @ X_world + t,
es decir T_cw (el MUNDO visto desde la CÁMARA). Para ubicar la cámara en el
mundo hay que INVERTIR:  R_wc = R^T ,  t_wc = -R^T t.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config import CameraParams, WorldMarker, marker_local_corners


@dataclass
class CameraPose:
    R_wc: np.ndarray            # rotación cámara -> mundo (3x3)
    t_wc: np.ndarray            # posición de la cámara en el mundo (3,) [m]
    reprojection_error: float   # error medio en píxeles
    markers_used: list[int]

    @property
    def quaternion_xyzw(self) -> np.ndarray:
        return rotation_to_quaternion(self.R_wc)


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Matriz de rotación -> cuaternión (x, y, z, w), formato de Foxglove."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class PoseEstimator:
    def __init__(self, cam: CameraParams, world: dict[int, WorldMarker],
                 marker_length: float, max_reprojection_error: float = 4.0):
        self.cam = cam
        self.world = world
        self.local_corners = marker_local_corners(marker_length)
        self.max_err = max_reprojection_error
        self._last: CameraPose | None = None

    # ------------------------------------------------------------ utilidades
    def _reproj_error(self, obj, img, rvec, tvec) -> float:
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.cam.K, self.cam.dist)
        return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))

    @staticmethod
    def _invert(R_cw: np.ndarray, t_cw: np.ndarray):
        return R_cw.T, -R_cw.T @ t_cw

    # ----------------------------------------------------------- un marcador
    def _solve_single(self, mid: int, img: np.ndarray):
        """IPPE_SQUARE: método específico para un cuadrado plano.

        Devuelve 2 soluciones (ambigüedad de pose planar). Elegimos la de menor
        error; si son parecidas, la más cercana a la pose anterior (evita "flips").
        """
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            self.local_corners, img, self.cam.K, self.cam.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if n == 0:
            return None
        m = self.world[mid]
        candidates = []
        for rvec, tvec in zip(rvecs, tvecs):
            R_cm, _ = cv2.Rodrigues(rvec)
            # Componer: T_cw = T_cm · T_mw,  con T_mw = inversa(T_wm)
            R_cw = R_cm @ m.R.T
            t_cw = tvec.reshape(3) - R_cw @ m.t
            err = self._reproj_error(self.local_corners, img, rvec, tvec)
            candidates.append((err, R_cw, t_cw))
        candidates.sort(key=lambda c: c[0])
        best = candidates[0]
        if self._last is not None and len(candidates) > 1 and candidates[1][0] < 1.5 * best[0]:
            best = min(candidates[:2], key=lambda c: np.linalg.norm(
                self._invert(c[1], c[2])[1] - self._last.t_wc))
        return best

    # ------------------------------------------------------ varios marcadores
    def _solve_multi(self, obj: np.ndarray, img: np.ndarray):
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.cam.K, self.cam.dist,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        # Refinamiento no lineal (Levenberg-Marquardt) minimizando la reproyección
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.cam.K, self.cam.dist, rvec, tvec)
        R_cw, _ = cv2.Rodrigues(rvec)
        return self._reproj_error(obj, img, rvec, tvec), R_cw, tvec.reshape(3)

    # ------------------------------------------------------------------- API
    def estimate(self, detections: dict[int, np.ndarray]) -> CameraPose | None:
        known = [i for i in sorted(detections) if i in self.world]
        if not known:
            return None
        if len(known) == 1:
            result = self._solve_single(known[0], detections[known[0]])
        else:
            obj = np.vstack([self.world[i].corners for i in known])
            img = np.vstack([detections[i] for i in known])
            result = self._solve_multi(obj, img)
        if result is None:
            return None
        err, R_cw, t_cw = result
        if err > self.max_err:
            return None
        R_wc, t_wc = self._invert(R_cw, t_cw)
        pose = CameraPose(R_wc, t_wc, err, known)
        self._last = pose
        return pose

    def draw_axes(self, frame, pose: CameraPose, length: float = 0.1) -> None:
        """Dibuja los ejes del MUNDO (origen) sobre la imagen. Validación visual."""
        R_cw, t_cw = self._invert(pose.R_wc, pose.t_wc)
        rvec, _ = cv2.Rodrigues(R_cw)
        cv2.drawFrameAxes(frame, self.cam.K, self.cam.dist, rvec, t_cw, length)
