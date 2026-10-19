import argparse
import sys
from pathlib import Path

import numpy as np
import cv2 as cv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_settings, resolve, save_camera_params  # noqa: E402

camCfg = load_settings()["camera"]
ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=camCfg["index"], help="índice de cámara (0 = default)")
ap.add_argument("--cols", type=int, default=9, help="esquinas internas horizontales")
ap.add_argument("--rows", type=int, default=6, help="esquinas internas verticales")
ap.add_argument("--square", type=float, default=1.0, help="lado del cuadrado [m]")
ap.add_argument("--images", default="calibration/images", help="carpeta donde guardar las fotos")
ap.add_argument("--out", default=camCfg["calibration_file"], help="archivo .yml de salida")
args = ap.parse_args()

imagesDir = resolve(args.images)
imagesDir.mkdir(parents=True, exist_ok=True)
# Numerar a continuación de las fotos que ya hay, para no sobrescribirlas
existing = [int(f.stem.split("_")[-1]) for f in imagesDir.glob("calib_*.png") if f.stem.split("_")[-1].isdigit()]
nextImage = max(existing, default=0) + 1

print("""
      Usage:
      Space: take picture
      C: calibrate with taken pictures
      ESC: quit
      """)

ESC = chr(27)
cv.namedWindow("Detecciones", cv.WINDOW_NORMAL)
cv.namedWindow("Cam", cv.WINDOW_NORMAL)
cv.namedWindow("Tablero", cv.WINDOW_NORMAL)
defaultPrintOptions = np.get_printoptions()

chessBoard = (args.cols, args.rows)
gradualDarkness = 0.90
cornersubpixTerminationCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
imgPoints = []
objPoints = []

# chessboard 3D points
chessboardPointCloud3D = np.zeros((chessBoard[0]*chessBoard[1],3), np.float32)
chessboardPointCloud3D[:,:2] = np.mgrid[0:chessBoard[0],0:chessBoard[1]].T.reshape(-1,2) * args.square

# La imagen se busca junto al script, no en la carpeta desde donde se ejecuta
imChessboard = cv.imread(str(Path(__file__).resolve().parent / "pattern_chessboard6x9.png"))
if imChessboard is not None:
    cv.imshow("Tablero", imChessboard)

camIndex = args.cam
cam = cv.VideoCapture(camIndex)
if not cam.isOpened():
    sys.exit(f"No se pudo abrir la cámara {camIndex}")
# Misma resolución que usa src.main (config/settings.yml): K solo vale para esa resolución
cam.set(cv.CAP_PROP_FRAME_WIDTH, camCfg["width"])
cam.set(cv.CAP_PROP_FRAME_HEIGHT, camCfg["height"])
cam.set(cv.CAP_PROP_AUTOFOCUS, 0)
width = cam.get(cv.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv.CAP_PROP_FRAME_HEIGHT)
print("Cam resolution:", width, " x ", height)
newSize = (640, int(640 * height / width))
imBlack = np.zeros(newSize[::-1]+(3,), dtype=np.uint8)

while True:
    ret, im = cam.read()
    if ret:
        imLowRes = cv.resize(im, newSize)
        imGrayLowRes = cv.cvtColor(imLowRes, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(imGrayLowRes, chessBoard, None)
        if ret:
            cv.drawChessboardCorners(imLowRes, chessBoard, corners, ret)
    
    cv.imshow('Cam', imLowRes)

    key = cv.waitKey(33)
    if key>=0:
        key = chr(key)
        print(key)
        match key:
            case ' ':
                # Repite la detección en alta resolución y la registra
                imGray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                ret, precisionCorners = cv.findChessboardCorners(imGray, chessBoard, None)
                if ret:
                    precisionCorners = cv.cornerSubPix(imGray, precisionCorners, (11,11), (-1,-1), cornersubpixTerminationCriteria)
                    imgPoints.append(precisionCorners)
                    objPoints.append(chessboardPointCloud3D)
                    imPath = imagesDir / f"calib_{nextImage:03d}.png"
                    cv.imwrite(str(imPath), im)
                    nextImage += 1

                    # Anota en baja resolución
                    imBlack = cv.convertScaleAbs(imBlack, alpha=gradualDarkness, beta=0)
                    cv.drawChessboardCorners(imBlack, chessBoard, corners, ret)
                    cv.imshow("Detecciones", imBlack)

                    print(len(imgPoints), "pictures taken ->", imPath)

            case 'c':
                # Calibra
                if not imgPoints:
                    print("Primero tomá fotos con ESPACIO")
                    continue
                ret, K, distCoef, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, im.shape[:2][::-1], None, None, flags=cv.CALIB_ZERO_TANGENT_DIST)

                # Muestra resultados
                print("Coeficientes de distorsión (K1, K2, P1, P2, K3):", distCoef)

                np.set_printoptions(precision=2, suppress=True)
                print("Matriz K", K)
                np.set_printoptions(**defaultPrintOptions)

                print(f"RMS de reproyección: {ret:.4f} px ({'OK' if ret < 0.5 else 'ALTO'})")
                h, w = im.shape[:2]
                save_camera_params(args.out, K, distCoef, w, h, ret)
                print("Guardado en", resolve(args.out))

            case ESC:
                print("Terminando.")
                break