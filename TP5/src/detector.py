"""Fase 2 - Detección de marcadores ArUco."""
from __future__ import annotations

import cv2
import numpy as np


def get_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Diccionario ArUco desconocido: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


class ArucoDetector:
    def __init__(self, dictionary_name: str = "DICT_4X4_50"):
        params = cv2.aruco.DetectorParameters()
        # Refinamiento subpíxel de esquinas: mejora bastante la precisión de PnP.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(get_dictionary(dictionary_name), params)

    def detect(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        """Devuelve {id: esquinas 4x2 (float64)} de los marcadores detectados."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _rejected = self._detector.detectMarkers(gray)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.flatten(), corners)}

    @staticmethod
    def draw(frame: np.ndarray, detections: dict[int, np.ndarray]) -> None:
        if not detections:
            return
        ids = np.array(list(detections.keys()), dtype=np.int32).reshape(-1, 1)
        corners = [c.reshape(1, 4, 2).astype(np.float32) for c in detections.values()]
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)


if __name__ == "__main__":
    # Prueba rápida de la Fase 2: python -m src.detector
    from src.config import load_settings

    cfg = load_settings()
    det = ArucoDetector(cfg["aruco"]["dictionary"])
    cap = cv2.VideoCapture(cfg["camera"]["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        d = det.detect(frame)
        det.draw(frame, d)
        cv2.putText(frame, f"marcadores: {sorted(d)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Fase 2 - Deteccion ArUco (q para salir)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
