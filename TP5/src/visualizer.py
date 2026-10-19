"""Fase 4 - Visualización 3D de la localización con Open3D.

Ventana "TP5 - Mundo 3D" con:
  - ejes del mundo (X rojo, Y verde, Z azul) y una grilla sobre la mesa (z = 0)
  - los marcadores del mapa (cuadrados oscuros; la línea roja marca el borde
    superior del marcador, o sea hacia dónde apunta su +Y local)
  - la cámara: sus ejes (convención OpenCV, Z azul = hacia donde mira) y su
    frustum calculado con los intrínsecos K
  - la trayectoria de la cámara (últimas N poses)

Mouse en la ventana: arrastrar = rotar, rueda = zoom, Ctrl+arrastrar = desplazar.
Cerrar la ventana termina el programa.

Si se pasa record_path, cada pose se agrega a un CSV (t, x, y, z, qx, qy, qz, qw, ...).
"""
from __future__ import annotations

import csv
import time
from collections import deque

import numpy as np
import open3d as o3d

from src.config import CameraParams, WorldMarker
from src.pose_estimator import CameraPose

FRUSTUM_DEPTH = 0.08     # profundidad del frustum dibujado (m)
CAMERA_AXES_SIZE = 0.05
GRID_STEP = 0.10


def _line_set(points, lines, color) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)),
        o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)))
    ls.paint_uniform_color(color)
    return ls


def _frustum_points(cam: CameraParams | None, width: int, height: int) -> np.ndarray:
    """Vértice + 4 esquinas de la imagen retroproyectadas a FRUSTUM_DEPTH (sistema cámara)."""
    if cam is not None:
        fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
    else:                                   # sin calibración: ~60° de FOV horizontal
        fx = fy = width / (2 * np.tan(np.deg2rad(30)))
        cx, cy = width / 2, height / 2
    d = FRUSTUM_DEPTH
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    pts = [[0, 0, 0]] + [[(u - cx) / fx * d, (v - cy) / fy * d, d] for u, v in corners]
    return np.array(pts, dtype=np.float64)


class Open3DVisualizer:
    def __init__(self, world: dict[int, WorldMarker], marker_length: float,
                 cam: CameraParams | None = None, image_size: tuple[int, int] = (640, 480),
                 trajectory_length: int = 300, record_path: str | None = None):
        self.trajectory: deque[np.ndarray] = deque(maxlen=trajectory_length)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("TP5 - Mundo 3D", width=960, height=720)
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.12, 0.12, 0.14])
        opt.point_size = 3.0

        # ---- escena fija
        centers = np.array([m.t for m in world.values()]) if world else np.zeros((1, 3))
        self._add_grid(centers, marker_length)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10))
        for m in world.values():
            self._add_marker(m, marker_length)

        # ---- cámara (se mueve en cada update). Se guarda la geometría en el
        # sistema de la cámara y en cada pose se transforma al mundo.
        self._axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=CAMERA_AXES_SIZE)
        self._axes_local = np.asarray(self._axes.vertices).copy()
        self._frustum_local = _frustum_points(cam, *image_size)
        self._frustum = _line_set(
            self._frustum_local,
            [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]],
            (1.0, 0.85, 0.1))
        self._traj = o3d.geometry.LineSet()
        self._traj_points = o3d.geometry.PointCloud()
        # Se agregan a la ventana recién con la primera pose: una geometría vacía
        # rompe el shader de líneas de Open3D y deja de dibujar el resto.
        self._camera_added = False

        ctr = self.vis.get_view_control()
        # Planos de recorte fijos: si no, Open3D los calcula con el bounding box de la
        # escena fija (plana, z = 0) y recorta la cámara cuando está por encima de la
        # mesa. Tienen que ir ANTES de set_zoom, que es el que recalcula la proyección.
        ctr.set_constant_z_near(0.005)
        ctr.set_constant_z_far(50.0)
        # Vista inicial: en diagonal, mirando un poco por encima del centro de los
        # marcadores, con margen para ver a la cámara arriba de la mesa
        ctr.set_lookat(centers.mean(axis=0) + np.array([0.0, 0.0, 0.15]))
        ctr.set_front([-0.5, -0.7, 0.5])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(1.4)

        self._csv_file = None
        if record_path:
            self._csv_file = open(record_path, "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(["t", "x", "y", "z", "qx", "qy", "qz", "qw",
                                "reproj_px", "markers"])
            self._t0 = time.time()
            print(f"[open3d] grabando poses en {record_path}")

    # ------------------------------------------------------------- escena fija
    def _add_grid(self, centers: np.ndarray, marker_length: float) -> None:
        margin = 0.3
        lo = np.floor((centers[:, :2].min(axis=0) - margin) / GRID_STEP) * GRID_STEP
        hi = np.ceil((centers[:, :2].max(axis=0) + margin) / GRID_STEP) * GRID_STEP
        pts, lines = [], []
        for x in np.arange(lo[0], hi[0] + 1e-9, GRID_STEP):
            pts += [[x, lo[1], 0], [x, hi[1], 0]]
            lines.append([len(pts) - 2, len(pts) - 1])
        for y in np.arange(lo[1], hi[1] + 1e-9, GRID_STEP):
            pts += [[lo[0], y, 0], [hi[0], y, 0]]
            lines.append([len(pts) - 2, len(pts) - 1])
        self.vis.add_geometry(_line_set(pts, lines, (0.3, 0.3, 0.33)))

    def _add_marker(self, m: WorldMarker, length: float) -> None:
        thick = 0.002
        box = o3d.geometry.TriangleMesh.create_box(length, length, thick)
        box.translate((-length / 2, -length / 2, -thick))   # centrado, cara superior en z=0
        box.rotate(m.R, center=(0, 0, 0))
        box.translate(m.t)
        box.compute_vertex_normals()
        box.paint_uniform_color((0.85, 0.85, 0.85))
        self.vis.add_geometry(box)
        # contorno + borde superior (esquinas 0 y 1 en el orden de OpenCV)
        c = m.corners + np.array([0, 0, 0.001])
        self.vis.add_geometry(_line_set(c, [[1, 2], [2, 3], [3, 0]], (0.05, 0.05, 0.05)))
        self.vis.add_geometry(_line_set(c, [[0, 1]], (0.9, 0.1, 0.1)))

    # ---------------------------------------------------------------- pose
    def update_pose(self, pose: CameraPose) -> None:
        R, t = pose.R_wc, pose.t_wc
        self._axes.vertices = o3d.utility.Vector3dVector(self._axes_local @ R.T + t)
        self._frustum.points = o3d.utility.Vector3dVector(self._frustum_local @ R.T + t)
        self.trajectory.append(t.copy())
        if not self._camera_added:
            # dos puntos iguales para que la trayectoria nunca quede vacía
            self.trajectory.append(t.copy())
            self._update_trajectory(redraw=False)
            for g in (self._axes, self._frustum, self._traj, self._traj_points):
                self.vis.add_geometry(g, reset_bounding_box=False)
            self._camera_added = True
        else:
            self.vis.update_geometry(self._axes)
            self.vis.update_geometry(self._frustum)
            self._update_trajectory()

        if self._csv_file is not None:
            q = pose.quaternion_xyzw
            self._csv.writerow([f"{time.time() - self._t0:.3f}", *(f"{v:.4f}" for v in t),
                                *(f"{v:.5f}" for v in q), f"{pose.reprojection_error:.3f}",
                                " ".join(map(str, pose.markers_used))])

    def clear_trajectory(self) -> None:
        # se conserva la última posición (duplicada) para no dejar la geometría vacía
        last = self.trajectory[-1] if self.trajectory else None
        self.trajectory.clear()
        if last is not None:
            self.trajectory.extend([last, last])
            self._update_trajectory()

    def _update_trajectory(self, redraw: bool = True) -> None:
        pts = np.array(self.trajectory)
        n = len(pts)
        self._traj.points = o3d.utility.Vector3dVector(pts)
        self._traj.lines = o3d.utility.Vector2iVector(
            np.column_stack([np.arange(n - 1), np.arange(1, n)]))
        self._traj.paint_uniform_color((0.1, 0.6, 1.0))
        self._traj_points.points = o3d.utility.Vector3dVector(pts)
        self._traj_points.paint_uniform_color((0.1, 0.6, 1.0))
        if redraw:
            self.vis.update_geometry(self._traj)
            self.vis.update_geometry(self._traj_points)

    # ---------------------------------------------------------------- loop
    def render(self) -> bool:
        """Procesa eventos y redibuja. Devuelve False si se cerró la ventana."""
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        self.vis.destroy_window()
