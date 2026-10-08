"""Test sintético de punta a punta (no necesita cámara ni calibración).

Se "fotografía" el mapa de marcadores desde una pose de cámara CONOCIDA
(renderizando con warpPerspective), se detecta, se estima la pose con PnP y
se compara con la verdadera.

Uso (desde la carpeta TP5):
    python -m tests.test_pose_synthetic
"""
import cv2
import numpy as np

from src.config import CameraParams, load_settings, load_world_map
from src.detector import ArucoDetector, get_dictionary
from src.pose_estimator import PoseEstimator
from tools.fake_pose_publisher import look_at

W, H = 1280, 720
K = np.array([[900.0, 0, W / 2], [0, 900.0, H / 2], [0, 0, 1]])
DIST = np.zeros((1, 5))


def render(world, dictionary, L, R_wc, t_wc) -> np.ndarray:
    img = np.full((H, W), 200, np.uint8)
    R_cw, t_cw = R_wc.T, -R_wc.T @ t_wc
    rvec, _ = cv2.Rodrigues(R_cw)
    px = 400
    for m in world.values():
        # Marcador con borde blanco (quiet zone), como en la impresión real
        tex = np.full((px * 2, px * 2), 255, np.uint8)
        tex[px // 2: px // 2 + px, px // 2: px // 2 + px] = cv2.aruco.generateImageMarker(dictionary, m.id, px)
        h = L  # la textura completa mide 2L (marcador L + márgenes)
        local = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        pts3d = (m.R @ local.T).T + m.t
        dst, _ = cv2.projectPoints(pts3d, rvec, t_cw, K, DIST)
        src = np.float32([[0, 0], [2 * px, 0], [2 * px, 2 * px], [0, 2 * px]])
        Hm = cv2.getPerspectiveTransform(src, dst.reshape(4, 2).astype(np.float32))
        warped = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(tex, 255), Hm, (W, H), borderValue=0)
        img[mask > 0] = warped[mask > 0]
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main():
    cfg = load_settings()
    L = cfg["aruco"]["marker_length"]
    world = load_world_map(cfg["world"]["map_file"], L)
    dictionary = get_dictionary(cfg["aruco"]["dictionary"])
    det = ArucoDetector(cfg["aruco"]["dictionary"])
    cam = CameraParams(K, DIST, W, H)
    center = np.mean([m.t for m in world.values()], axis=0)

    cases = [np.array([0.15, -0.35, 0.45]), np.array([0.6, 0.1, 0.5]),
             np.array([-0.2, 0.5, 0.35]), np.array([0.15, 0.1, 0.7])]
    worst = 0.0
    for single in (False, True):
        w = {0: world[0]} if single else world
        tgt = world[0].t if single else center
        est = PoseEstimator(cam, w, L, max_reprojection_error=5.0)
        for eye in cases:
            R_true = look_at(eye.copy(), tgt)
            img = render(w, dictionary, L, R_true, eye)
            pose = est.estimate(det.detect(img))
            assert pose is not None, f"sin pose para {eye}"
            pos_err = np.linalg.norm(pose.t_wc - eye)
            ang_err = np.degrees(np.arccos(np.clip((np.trace(pose.R_wc.T @ R_true) - 1) / 2, -1, 1)))
            worst = max(worst, pos_err)
            print(f"{'1 marcador ' if single else 'multi      '} real={eye} est={np.round(pose.t_wc, 4)} "
                  f"err={pos_err * 1000:.1f}mm ang={ang_err:.2f}° reproj={pose.reprojection_error:.2f}px "
                  f"ids={pose.markers_used}")
    assert worst < 0.02, f"error máximo {worst:.3f} m"
    print(f"\nOK - error máximo de posición: {worst * 1000:.1f} mm")


if __name__ == "__main__":
    main()
