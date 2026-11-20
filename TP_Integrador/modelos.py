"""
Envoltorios (wrappers) uniformes para TwinLiteNetPlus y Ultra-Fast-Lane-Detection-v2.

Cada modelo expone:
    m = Modelo(...)
    salida = m.inferir(frame_bgr)      # frame tal cual sale de cv2.VideoCapture
    vis    = m.dibujar(frame_bgr, salida)
    xl, xr = m.bordes_carril(salida, y)  # x del borde izq./der. del carril propio en la fila y (o None)

Así el script de comparación trata a los dos modelos de la misma forma.
"""
import os
import sys
import types
import importlib
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Utilidad: los dos repos tienen un paquete llamado "model" (y UFLD uno "utils"),
# así que se importan de a uno, limpiando sys.modules entre medio.
# ----------------------------------------------------------------------------
_CONFLICTIVOS = ("model", "utils", "data")


def _importar_desde_repo(repo, modulos, stubs=None):
    repo = os.path.abspath(repo)
    guardados = {k: v for k, v in sys.modules.items()
                 if k.split(".")[0] in _CONFLICTIVOS}
    for k in guardados:
        del sys.modules[k]
    sys.path.insert(0, repo)
    try:
        for nombre, mod in (stubs or {}).items():
            sys.modules[nombre] = mod
        return [importlib.import_module(m) for m in modulos]
    finally:
        sys.path.remove(repo)
        for k in [k for k in sys.modules if k.split(".")[0] in _CONFLICTIVOS]:
            del sys.modules[k]
        sys.modules.update(guardados)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def contar_parametros(net):
    return sum(p.numel() for p in net.parameters())


# ----------------------------------------------------------------------------
# TwinLiteNetPlus  (segmentación: área manejable + líneas de carril)
# ----------------------------------------------------------------------------
def _letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Mismo letterbox que usa demoDataset.py del repo (auto=True)."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = np.mod(dw, 32) / 2, np.mod(dh, 32) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, (top, bottom, left, right)


class TwinLite:
    nombre = "TwinLiteNetPlus"

    def __init__(self, repo, config="large", weight=None, device="cpu",
                 img_size=640, half=False):
        self.device = torch.device(device)
        self.img_size = img_size
        self.half = half and self.device.type == "cuda"
        (mm,) = _importar_desde_repo(repo, ["model.model"])
        args = types.SimpleNamespace(config=config)
        self.net = mm.TwinLiteNetPlus(args)
        if weight:
            sd = torch.load(weight, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            self.net.load_state_dict(sd)
        self.net.to(self.device).eval()
        if self.half:
            self.net.half()
        self.variante = config
        self.params = contar_parametros(self.net)

    @torch.inference_mode()
    def inferir(self, frame):
        h0, w0 = frame.shape[:2]
        t0 = time.perf_counter()
        img, (top, bottom, left, right) = _letterbox(frame, self.img_size)
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))
        x = x.unsqueeze(0).to(self.device)
        x = x.half() / 255.0 if self.half else x.float() / 255.0
        _sync(self.device)
        t1 = time.perf_counter()
        da, ll = self.net(x)
        _sync(self.device)
        t2 = time.perf_counter()
        H, W = x.shape[2:]
        da = da[0, :, top:H - bottom, left:W - right].argmax(0).byte().cpu().numpy()
        ll = ll[0, :, top:H - bottom, left:W - right].argmax(0).byte().cpu().numpy()
        da = cv2.resize(da, (w0, h0), interpolation=cv2.INTER_NEAREST)
        ll = cv2.resize(ll, (w0, h0), interpolation=cv2.INTER_NEAREST)
        t3 = time.perf_counter()
        return {"da": da, "ll": ll,
                "t_modelo_ms": (t2 - t1) * 1e3,
                "t_total_ms": (t3 - t0) * 1e3}

    def dibujar(self, frame, out):
        vis = frame.copy()
        color = np.zeros_like(vis)
        color[out["da"] == 1] = (0, 200, 0)     # área manejable: verde
        color[out["ll"] == 1] = (0, 0, 255)     # líneas de carril: rojo
        m = color.any(2)
        vis[m] = (0.5 * vis[m] + 0.5 * color[m]).astype(np.uint8)
        return vis

    def bordes_carril(self, out, y):
        """Busca, en la fila y, la línea más cercana al centro a cada lado."""
        fila = out["ll"][int(y)]
        xs = np.flatnonzero(fila)
        cx = fila.shape[0] / 2
        izq, der = xs[xs < cx], xs[xs >= cx]
        xl = float(izq.max()) if izq.size else None
        xr = float(der.min()) if der.size else None
        return xl, xr

    def cobertura(self, out):
        return {"pct_area_manejable": 100 * float(out["da"].mean()),
                "pct_lineas": 100 * float(out["ll"].mean())}


# ----------------------------------------------------------------------------
# Ultra-Fast-Lane-Detection v2  (detección de carriles por anclas)
# ----------------------------------------------------------------------------
class UFLD:
    nombre = "UFLDv2"
    COLORES = [(255, 128, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]

    def __init__(self, repo, config, weight=None, device="cpu"):
        self.device = torch.device(device)
        # utils/common.py importa NVIDIA DALI (solo se usa para entrenar): lo simulamos.
        stub = types.ModuleType("data.dali_data")
        stub.TrainCollect = None
        stubs = {"data.dali_data": stub}
        try:
            importlib.import_module("torch.utils.tensorboard")  # también solo para entrenar
        except Exception:
            tb = types.ModuleType("torch.utils.tensorboard")
            tb.SummaryWriter = object
            stubs["torch.utils.tensorboard"] = tb
        cfgmod, mc = _importar_desde_repo(
            repo, ["utils.config", "model.model_culane"], stubs=stubs)
        cfg = cfgmod.Config.fromfile(config)
        self.cfg = cfg
        self.net = mc.parsingNet(
            pretrained=False, backbone=cfg.backbone,
            num_grid_row=cfg.num_cell_row, num_cls_row=cfg.num_row,
            num_grid_col=cfg.num_cell_col, num_cls_col=cfg.num_col,
            num_lane_on_row=cfg.num_lanes, num_lane_on_col=cfg.num_lanes,
            use_aux=False, input_height=cfg.train_height,
            input_width=cfg.train_width, fc_norm=cfg.fc_norm)
        if weight:
            sd = torch.load(weight, map_location="cpu")
            sd = sd.get("model", sd)
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            self.net.load_state_dict(sd, strict=False)
        self.net.to(self.device).eval()
        if cfg.dataset == "CULane":
            self.row_anchor = np.linspace(0.42, 1, cfg.num_row)
        elif cfg.dataset == "Tusimple":
            self.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
        else:
            raise NotImplementedError("Solo configs CULane o Tusimple")
        self.col_anchor = np.linspace(0, 1, cfg.num_col)
        self.variante = f"{cfg.dataset}-res{cfg.backbone}"
        self.params = contar_parametros(self.net)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def inferir(self, frame):
        h0, w0 = frame.shape[:2]
        cfg = self.cfg
        t0 = time.perf_counter()
        # Igual que demo.py: resize a (train_h/crop_ratio, train_w) y quedarse con la parte de abajo
        hh = int(cfg.train_height / cfg.crop_ratio)
        img = cv2.resize(frame, (cfg.train_width, hh), interpolation=cv2.INTER_LINEAR)
        img = img[hh - cfg.train_height:, :, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0)
        x = (x.to(self.device).float() / 255.0 - self.mean) / self.std
        _sync(self.device)
        t1 = time.perf_counter()
        pred = self.net(x)
        _sync(self.device)
        t2 = time.perf_counter()
        carriles = self._pred2coords(pred, w0, h0)
        t3 = time.perf_counter()
        return {"carriles": carriles,
                "t_modelo_ms": (t2 - t1) * 1e3,
                "t_total_ms": (t3 - t0) * 1e3}

    def _pred2coords(self, pred, img_w, img_h, local_width=1):
        """Adaptado de demo.py del repo. Devuelve {id_carril: [(x, y), ...]}.
        id 1 y 2 = bordes izq./der. del carril propio; 0 y 3 = carriles vecinos."""
        _, ngr, ncr, _ = pred["loc_row"].shape
        _, ngc, ncc, _ = pred["loc_col"].shape
        loc_row, loc_col = pred["loc_row"].cpu(), pred["loc_col"].cpu()
        max_r, val_r = loc_row.argmax(1), pred["exist_row"].argmax(1).cpu()
        max_c, val_c = loc_col.argmax(1), pred["exist_col"].argmax(1).cpu()
        out = {}
        for i in (1, 2):
            if val_r[0, :, i].sum() > ncr / 2:
                pts = []
                for k in range(ncr):
                    if val_r[0, k, i]:
                        ind = torch.arange(max(0, max_r[0, k, i] - local_width),
                                           min(ngr - 1, max_r[0, k, i] + local_width) + 1)
                        x = (loc_row[0, ind, k, i].softmax(0) * ind.float()).sum() + 0.5
                        pts.append((int(x / (ngr - 1) * img_w), int(self.row_anchor[k] * img_h)))
                out[i] = pts
        for i in (0, 3):
            if val_c[0, :, i].sum() > ncc / 4:
                pts = []
                for k in range(ncc):
                    if val_c[0, k, i]:
                        ind = torch.arange(max(0, max_c[0, k, i] - local_width),
                                           min(ngc - 1, max_c[0, k, i] + local_width) + 1)
                        y = (loc_col[0, ind, k, i].softmax(0) * ind.float()).sum() + 0.5
                        pts.append((int(self.col_anchor[k] * img_w), int(y / (ngc - 1) * img_h)))
                out[i] = pts
        return out

    def dibujar(self, frame, out):
        vis = frame.copy()
        r = max(2, frame.shape[1] // 300)
        for i, pts in out["carriles"].items():
            for p in pts:
                cv2.circle(vis, p, r, self.COLORES[i], -1)
        return vis

    def bordes_carril(self, out, y):
        """Interpola la x de los carriles 1 (izq.) y 2 (der.) en la fila y."""
        res = []
        for i in (1, 2):
            pts = out["carriles"].get(i)
            if not pts or len(pts) < 2:
                res.append(None)
                continue
            pts = sorted(pts, key=lambda p: p[1])
            ys = np.array([p[1] for p in pts], float)
            xs = np.array([p[0] for p in pts], float)
            if y < ys.min() - 0.05 * ys.max() or y > ys.max() + 0.05 * ys.max():
                res.append(None)
            else:
                res.append(float(np.interp(y, ys, xs)))
        return tuple(res)

    def cobertura(self, out):
        return {"n_carriles": len(out["carriles"])}
