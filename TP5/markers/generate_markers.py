"""Fase 2a - Generación de marcadores ArUco para imprimir.

Uso (desde la carpeta TP5):
    python markers/generate_markers.py                 # IDs del world_map.yml
    python markers/generate_markers.py --ids 0 1 2 3

Genera en markers/output/:
  - aruco_<id>.png       un marcador por archivo
  - hoja_A4_<id>.png     hoja A4 a 300 dpi con el marcador al tamaño REAL
                         (marker_length de config/settings.yml)

IMPORTANTE: imprimir al 100% / "tamaño real" (sin "ajustar a página") y
después MEDIR con regla. Si no da exacto, poner la medida real en
settings.yml -> aruco.marker_length.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_settings, resolve  # noqa: E402
from src.detector import get_dictionary  # noqa: E402

DPI = 300
A4_MM = (210, 297)


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def main():
    cfg = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--length", type=float, default=cfg["aruco"]["marker_length"],
                    help="lado del marcador en metros")
    ap.add_argument("--dict", default=cfg["aruco"]["dictionary"])
    args = ap.parse_args()

    ids = args.ids
    if not ids:
        with open(resolve(cfg["world"]["map_file"]), encoding="utf-8") as f:
            ids = [m["id"] for m in yaml.safe_load(f)["markers"]]

    out = resolve("markers/output")
    out.mkdir(parents=True, exist_ok=True)
    dictionary = get_dictionary(args.dict)
    side_px = mm_to_px(args.length * 1000)
    page_w, page_h = mm_to_px(A4_MM[0]), mm_to_px(A4_MM[1])

    for i in ids:
        marker = cv2.aruco.generateImageMarker(dictionary, i, side_px)
        cv2.imwrite(str(out / f"aruco_{i}.png"), marker)

        page = np.full((page_h, page_w), 255, np.uint8)
        y0, x0 = (page_h - side_px) // 2, (page_w - side_px) // 2
        page[y0:y0 + side_px, x0:x0 + side_px] = marker
        label = f"{args.dict}  ID {i}  lado {args.length * 100:.1f} cm  (imprimir al 100%)"
        cv2.putText(page, label, (mm_to_px(15), mm_to_px(20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
        # Flecha indicando el borde superior del marcador (+Y del mundo con yaw 0)
        cv2.arrowedLine(page, (page_w // 2, y0 - mm_to_px(5)), (page_w // 2, y0 - mm_to_px(20)),
                        0, 6, tipLength=0.3)
        cv2.imwrite(str(out / f"hoja_A4_{i}.png"), page)
        print(f"ID {i}: {side_px}px = {args.length * 100:.1f} cm a {DPI} dpi")

    # PDF opcional (más confiable para imprimir a tamaño real)
    try:
        from PIL import Image
        pages = [Image.open(out / f"hoja_A4_{i}.png").convert("L") for i in ids]
        pdf = out / "marcadores_A4.pdf"
        pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=DPI)
        print("PDF:", pdf)
    except ImportError:
        print("(pip install pillow para generar también un PDF)")


if __name__ == "__main__":
    main()
