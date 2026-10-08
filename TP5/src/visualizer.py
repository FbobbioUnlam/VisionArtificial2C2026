"""Fase 4 - Publicación de la localización en Foxglove (SDK de Python).

Tópicos publicados:
  /tf                  FrameTransforms   world -> camera (lo que "mueve" la cámara)
  /world/markers       SceneUpdate       marcadores del mapa dibujados en el mundo
  /camera/trajectory   SceneUpdate       recorrido de la cámara (últimas N poses)
  /camera/calibration  CameraCalibration intrínsecos -> Foxglove dibuja el frustum
  /camera/image        CompressedImage   imagen con detecciones superpuestas

En Foxglove: Open connection -> Foxglove WebSocket -> ws://localhost:8765
y en el panel 3D elegir "world" como Display frame.
"""
from __future__ import annotations

from collections import deque

import cv2
import foxglove
import numpy as np
from foxglove.messages import (
    CameraCalibration, Color, CompressedImage, CubePrimitive, FrameTransform,
    FrameTransforms, LinePrimitive, LinePrimitiveLineType, Point3, Pose, Quaternion,
    SceneEntity, SceneUpdate, TextPrimitive, Vector3,
)

from src.config import CameraParams, WorldMarker
from src.pose_estimator import CameraPose, rotation_to_quaternion

WORLD_FRAME = "world"
CAMERA_FRAME = "camera"


def _quat(q) -> Quaternion:
    return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


def _vec(v) -> Vector3:
    return Vector3(x=float(v[0]), y=float(v[1]), z=float(v[2]))


class FoxgloveVisualizer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765,
                 trajectory_length: int = 300, mcap_path: str | None = None):
        self.server = foxglove.start_server(name="TP5 PnP", host=host, port=port)
        self.mcap = foxglove.open_mcap(mcap_path, allow_overwrite=True) if mcap_path else None
        self.trajectory: deque[np.ndarray] = deque(maxlen=trajectory_length)
        print(f"[foxglove] servidor en ws://{host}:{port}"
              + (f" | grabando en {mcap_path}" if mcap_path else ""))

    # ---------------------------------------------------------------- pose
    def publish_pose(self, pose: CameraPose) -> None:
        foxglove.log("/tf", FrameTransforms(transforms=[FrameTransform(
            parent_frame_id=WORLD_FRAME,
            child_frame_id=CAMERA_FRAME,
            translation=_vec(pose.t_wc),
            rotation=_quat(pose.quaternion_xyzw),
        )]))
        self.trajectory.append(pose.t_wc.copy())
        self._publish_trajectory()

    def _publish_trajectory(self) -> None:
        if len(self.trajectory) < 2:
            return
        line = LinePrimitive(
            type=LinePrimitiveLineType.LineStrip,
            pose=Pose(position=_vec((0, 0, 0)), orientation=_quat((0, 0, 0, 1))),
            thickness=3.0,
            scale_invariant=True,
            points=[Point3(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self.trajectory],
            color=Color(r=0.1, g=0.6, b=1.0, a=1.0),
        )
        foxglove.log("/camera/trajectory", SceneUpdate(entities=[
            SceneEntity(id="trajectory", frame_id=WORLD_FRAME, lines=[line])]))

    # ------------------------------------------------------------- escena fija
    def publish_world(self, world: dict[int, WorldMarker], marker_length: float) -> None:
        entities = []
        for m in world.values():
            q = rotation_to_quaternion(m.R)
            entities.append(SceneEntity(
                id=f"marker_{m.id}",
                frame_id=WORLD_FRAME,
                cubes=[CubePrimitive(
                    pose=Pose(position=_vec(m.t), orientation=_quat(q)),
                    size=_vec((marker_length, marker_length, 0.002)),
                    color=Color(r=0.15, g=0.15, b=0.15, a=1.0),
                )],
                texts=[TextPrimitive(
                    pose=Pose(position=_vec(m.t + np.array([0, 0, 0.03])), orientation=_quat((0, 0, 0, 1))),
                    billboard=True, font_size=14.0, scale_invariant=True,
                    color=Color(r=1.0, g=1.0, b=1.0, a=1.0), text=f"ID {m.id}",
                )],
            ))
        foxglove.log("/world/markers", SceneUpdate(entities=entities))

    def publish_calibration(self, cam: CameraParams, width: int, height: int) -> None:
        K = cam.K
        d = list(cam.dist.flatten()[:5]) + [0.0] * max(0, 5 - cam.dist.size)
        P = [K[0, 0], 0.0, K[0, 2], 0.0, 0.0, K[1, 1], K[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0]
        foxglove.log("/camera/calibration", CameraCalibration(
            frame_id=CAMERA_FRAME, width=width, height=height,
            distortion_model="plumb_bob", D=[float(x) for x in d],
            K=[float(x) for x in K.flatten()],
            R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            P=[float(x) for x in P],
        ))

    def publish_image(self, frame: np.ndarray, quality: int = 70) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            foxglove.log("/camera/image", CompressedImage(
                frame_id=CAMERA_FRAME, format="jpeg", data=buf.tobytes()))

    def close(self) -> None:
        if self.mcap is not None:
            self.mcap.close()
        self.server.stop()
