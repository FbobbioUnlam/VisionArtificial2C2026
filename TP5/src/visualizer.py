"""Fase 4 - Visualización 3D de la localización con Open3D.

Ventana "TP5 - Mundo 3D" con:
  - ejes del mundo (X rojo, Y verde, Z azul) y una grilla sobre la mesa (z = 0)
  - los marcadores del mapa con su ID escrito adelante (la línea roja marca el
    borde superior del marcador, o sea hacia dónde apunta su +Y local). Los
    marcadores detectados en el frame actual se pintan de verde.
  - la cámara: sus ejes (convención OpenCV, Z azul = hacia donde mira) y su
    frustum calculado con los intrínsecos K
  - la trayectoria de la cámara (últimas N poses)

Mouse en la ventana: arrastrar = rotar, rueda = zoom, Ctrl+arrastrar = desplazar.
Teclas en la ventana 3D:  F = encuadrar todo (marcadores + cámara + trayectoria),
                          A = activar/desactivar el encuadre automático.
Con el encuadre automático, la vista se reencuadra sola cuando la cámara sale de
la pantalla (manteniendo la dirección desde la que se está mirando).
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
MARKER_COLOR = (0.85, 0.85, 0.85)
DETECTED_COLOR = (0.35, 0.85, 0.35)
LABEL_COLOR = (0.75, 0.75, 0.75)
DETECTED_LABEL_COLOR = (0.35, 0.95, 0.35)


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
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("TP5 - Mundo 3D", width=960, height=720)
        self.auto_fit = True
        self.vis.register_key_callback(ord("F"), lambda _v: self.fit_view() or False)
        self.vis.register_key_callback(ord("A"), lambda _v: self._toggle_auto_fit() or False)
        print("[open3d] ventana 3D: F = encuadrar todo, A = encuadre automático on/off")
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.12, 0.12, 0.14])
        opt.point_size = 3.0

        opt.mesh_show_back_face = True

        # ---- escena fija
        self._marker_boxes: dict[int, o3d.geometry.TriangleMesh] = {}
        self._marker_labels: dict[int, o3d.geometry.TriangleMesh] = {}
        self._detected: set[int] = set()
        centers = np.array([m.t for m in world.values()]) if world else np.zeros((1, 3))
        self._static_pts = (np.vstack([m.corners for m in world.values()])
                            if world else np.zeros((1, 3)))
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
        box.paint_uniform_color(MARKER_COLOR)
        self.vis.add_geometry(box)
        self._marker_boxes[m.id] = box
        self._add_label(m, length)
        # contorno + borde superior (esquinas 0 y 1 en el orden de OpenCV)
        c = m.corners + np.array([0, 0, 0.001])
        self.vis.add_geometry(_line_set(c, [[1, 2], [2, 3], [3, 0]], (0.05, 0.05, 0.05)))
        self.vis.add_geometry(_line_set(c, [[0, 1]], (0.9, 0.1, 0.1)))

    def _add_label(self, m: WorldMarker, length: float) -> None:
        """ID del marcador, apoyado en la mesa justo delante (-Y) del marcador."""
        # depth > 0: texto extruido (sólido); con depth = 0 solo se ve el contorno
        text = o3d.t.geometry.TriangleMesh.create_text(f"ID {m.id}", depth=1.0).to_legacy()
        bb = text.get_axis_aligned_bounding_box()
        text.translate(-bb.get_center())
        height = 0.4 * length
        text.scale(height / max(bb.get_extent()[1], 1e-9), center=(0, 0, 0))
        text.translate(m.t + np.array([0.0, -(0.5 * length + 0.7 * height), 0.002]))
        text.compute_vertex_normals()
        text.paint_uniform_color(LABEL_COLOR)
        self.vis.add_geometry(text)
        self._marker_labels[m.id] = text
        # el texto también tiene que entrar al encuadrar la vista
        self._static_pts = np.vstack([self._static_pts,
                                      np.asarray(text.get_axis_aligned_bounding_box().get_box_points())])

    def set_detected(self, ids) -> None:
        """Pinta de verde los marcadores detectados en el frame actual."""
        ids = set(ids) & self._marker_boxes.keys()
        if ids == self._detected:
            return
        for mid in ids ^ self._detected:            # solo los que cambiaron
            on = mid in ids
            self._marker_boxes[mid].paint_uniform_color(DETECTED_COLOR if on else MARKER_COLOR)
            self._marker_labels[mid].paint_uniform_color(DETECTED_LABEL_COLOR if on else LABEL_COLOR)
            self.vis.update_geometry(self._marker_boxes[mid])
            self.vis.update_geometry(self._marker_labels[mid])
        self._detected = ids

    # ---------------------------------------------------------------- pose
    def update_pose(self, pose: CameraPose) -> None:
        R, t = pose.R_wc, pose.t_wc
        # La cámara se dibuja más grande cuanto más lejos está de los marcadores,
        # para que siga siendo visible cuando la vista se aleja
        k = max(1.0, float(np.linalg.norm(t - self._static_pts.mean(axis=0))) / 0.6)
        self._axes.vertices = o3d.utility.Vector3dVector(k * self._axes_local @ R.T + t)
        self._frustum.points = o3d.utility.Vector3dVector(k * self._frustum_local @ R.T + t)
        self.trajectory.append(t.copy())
        if not self._camera_added:
            # dos puntos iguales para que la trayectoria nunca quede vacía
            self.trajectory.append(t.copy())
            self._update_trajectory(redraw=False)
            for g in (self._axes, self._frustum, self._traj, self._traj_points):
                self.vis.add_geometry(g, reset_bounding_box=False)
            self._camera_added = True
            self.fit_view()
        else:
            self.vis.update_geometry(self._axes)
            self.vis.update_geometry(self._frustum)
            self._update_trajectory()
            if self.auto_fit and not self._in_view(t):
                self.fit_view()

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

    # ---------------------------------------------------------------- vista
    def fit_view(self) -> None:
        """Acerca/aleja la vista para que entren marcadores, cámara y trayectoria.

        Mantiene la dirección desde la que se está mirando: solo cambia el punto
        al que se mira y la distancia, y se aplica como una cámara pinhole.
        """
        pts = [self._static_pts]
        if self._camera_added:
            pts += [np.asarray(self._frustum.points), np.array(self.trajectory)]
        pts = np.vstack(pts)
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2

        ctr = self.vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        Kv = params.intrinsic.intrinsic_matrix
        tan_x = params.intrinsic.width / (2 * Kv[0, 0])
        tan_y = params.intrinsic.height / (2 * Kv[1, 1])
        E = np.array(params.extrinsic)
        # Filas: derecha, abajo, adelante. Open3D guarda el vector "up" tal como se
        # le pasó, sin hacerlo perpendicular a la dirección de vista, así que la
        # rotación que devuelve no es ortonormal: se la reconstruye.
        fwd = E[2, :3] / np.linalg.norm(E[2, :3])
        right = E[0, :3] - (E[0, :3] @ fwd) * fwd
        right /= np.linalg.norm(right)
        R = np.vstack([right, np.cross(fwd, right), fwd])
        E[:3, :3] = R
        # Cada punto, en coordenadas de la vista y relativo al centro, exige una
        # distancia mínima para entrar en pantalla (con 10 % de margen):
        #   |x| / (d + z) <= 0.9 tan_x   y   |y| / (d + z) <= 0.9 tan_y
        q = (pts - center) @ R.T
        dist = max(float(np.max(np.abs(q[:, 0]) / (0.9 * tan_x) - q[:, 2])),
                   float(np.max(np.abs(q[:, 1]) / (0.9 * tan_y) - q[:, 2])), 0.2)
        eye = center - R[2] * dist
        E[:3, 3] = -R @ eye
        params.extrinsic = E
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    def _in_view(self, p: np.ndarray, margin: float = 0.05) -> bool:
        """True si el punto p (mundo) cae dentro de la ventana 3D, con un margen."""
        params = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        pc = params.extrinsic[:3, :3] @ p + params.extrinsic[:3, 3]
        if pc[2] <= 0:
            return False
        uv = (params.intrinsic.intrinsic_matrix @ (pc / pc[2]))[:2]
        w, h = params.intrinsic.width, params.intrinsic.height
        return (margin * w <= uv[0] <= (1 - margin) * w) and (margin * h <= uv[1] <= (1 - margin) * h)

    def _toggle_auto_fit(self) -> None:
        self.auto_fit = not self.auto_fit
        print(f"[open3d] encuadre automático: {'activado' if self.auto_fit else 'desactivado'}")
        if self.auto_fit:
            self.fit_view()

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
