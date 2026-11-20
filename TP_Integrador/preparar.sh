#!/usr/bin/env bash
# Prepara todo: clona los repos, instala dependencias y descarga los pesos preentrenados.
# Uso:  bash preparar.sh            (desde la carpeta del kit)
set -e
cd "$(dirname "$0")"

echo "== 1/3 Clonando repositorios =="
mkdir -p repos
[ -d repos/TwinLiteNetPlus ] || git clone --depth 1 https://github.com/chequanghuy/TwinLiteNetPlus repos/TwinLiteNetPlus
[ -d repos/ultra-fast-lane-detection-v2 ] || git clone --depth 1 https://github.com/cfzd/ultra-fast-lane-detection-v2 repos/ultra-fast-lane-detection-v2

echo "== 2/3 Instalando dependencias =="
# Si ya tenés PyTorch con CUDA instalado, esto no lo reemplaza.
python -c "import torch" 2>/dev/null || pip install torch torchvision
pip install -q opencv-python numpy pandas matplotlib tabulate addict tqdm gdown

echo "== 3/3 Descargando pesos (Google Drive) =="
mkdir -p pesos/twinlite pesos/ufld
# TwinLiteNetPlus: carpeta con nano/small/medium/large.pth (entrenados en BDD100K)
[ -n "$(ls pesos/twinlite 2>/dev/null)" ] || \
  gdown --folder "https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz" -O pesos/twinlite
# UFLDv2 (ResNet18 en CULane y TuSimple; descomentá ResNet34 si querés probarlo)
[ -f pesos/ufld/culane_res18.pth ]   || gdown 1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W -O pesos/ufld/culane_res18.pth
[ -f pesos/ufld/tusimple_res18.pth ] || gdown 1Clnj9-dLz81S3wXiYtlkc4HVusCb978t -O pesos/ufld/tusimple_res18.pth
# [ -f pesos/ufld/culane_res34.pth ] || gdown 1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj -O pesos/ufld/culane_res34.pth

echo
echo "Pesos descargados:"
find pesos -name "*.pth" | sort
echo
echo "Listo. Poné tus videos en videos/unlam y videos/bfmc y corré, por ejemplo:"
echo "  python comparar.py --videos videos/unlam videos/bfmc \\"
echo "     --tl-weight \$(find pesos/twinlite -name 'large.pth' | head -1) --tl-config large \\"
echo "     --ufld-weight pesos/ufld/culane_res18.pth --ufld-config culane_res18"
