"""Fase 5 - Suavizado de la pose (reduce el temblor/jitter)."""
from __future__ import annotations

import numpy as np

from src.pose_estimator import CameraPose, quaternion_to_rotation, rotation_to_quaternion


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Interpolación esférica entre cuaterniones (x, y, z, w)."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:              # q y -q son la misma rotación: tomar el camino corto
        q1, dot = -q1, -dot
    if dot > 0.9995:           # casi iguales: interpolación lineal normalizada
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


class PoseSmoother:
    """Filtro exponencial: posición con media móvil exponencial, rotación con slerp.

    alpha = 0 -> sin filtro; alpha cercano a 1 -> muy suave pero con retardo.
    """

    def __init__(self, alpha: float = 0.5, reset_distance: float = 0.5):
        self.alpha = float(np.clip(alpha, 0.0, 0.99))
        self.reset_distance = reset_distance  # si salta más que esto (m), no filtrar
        self._t: np.ndarray | None = None
        self._q: np.ndarray | None = None

    def update(self, pose: CameraPose) -> CameraPose:
        q = pose.quaternion_xyzw
        if self._t is None or np.linalg.norm(pose.t_wc - self._t) > self.reset_distance:
            self._t, self._q = pose.t_wc.copy(), q
        else:
            k = 1.0 - self.alpha
            self._t = self._t + k * (pose.t_wc - self._t)
            self._q = slerp(self._q, q, k)
        return CameraPose(quaternion_to_rotation(self._q), self._t.copy(),
                          pose.reprojection_error, pose.markers_used)

    @staticmethod
    def quaternion(pose: CameraPose) -> np.ndarray:
        return rotation_to_quaternion(pose.R_wc)
