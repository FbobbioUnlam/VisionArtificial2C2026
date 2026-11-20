#!/usr/bin/env python3
"""
Comparación de TwinLiteNetPlus vs Ultra-Fast-Lane-Detection-v2 sobre videos.

Ejemplo:
    python comparar.py --videos videos/unlam videos/bfmc --salida resultados \
        --tl-weight pesos/twinlite/large.pth --tl-config large \
        --ufld-weight pesos/ufld/culane_res18.pth --ufld-config culane_res18

Por cada video genera en <salida>/<grupo>/<video>/:
    comparacion.mp4   video lado a lado (izq. TwinLiteNetPlus, der. UFLDv2)
    frames.csv        métricas cuadro a cuadro de ambos modelos
    centro_carril.png posición estimada del centro del carril en el tiempo
    capturas/*.png    cuadros de muestra para el informe
y en <salida>/: resumen.csv, resumen.md y graficos_resumen.png
"""
import argparse
import os
import sys
import glob
import json

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelos import TwinLite, UFLD  # noqa: E402

EXT_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def listar_videos(rutas):
    vids = []
    for r in rutas:
        if os.path.isdir(r):
            for f in sorted(glob.glob(os.path.join(r, "**", "*"), recursive=True)):
                if f.lower().endswith(EXT_VIDEO):
                    vids.append((os.path.basename(os.path.normpath(r)), f))
        elif r.lower().endswith(EXT_VIDEO):
            vids.append((os.path.basename(os.path.dirname(os.path.abspath(r))), r))
    return vids


def rotulo(img, texto, sub=""):
    h = 34 if not sub else 54
    barra = np.zeros((h, img.shape[1], 3), np.uint8)
    cv2.putText(barra, texto, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(barra, sub, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([barra, img])


def marcar_fila(img, y, xl, xr):
    cv2.line(img, (0, int(y)), (img.shape[1], int(y)), (255, 255, 255), 1)
    for x in (xl, xr):
        if x is not None:
            cv2.circle(img, (int(x), int(y)), 7, (255, 255, 255), 2)
    if xl is not None and xr is not None:
        cv2.drawMarker(img, (int((xl + xr) / 2), int(y)), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    return img


def procesar_video(grupo, ruta, modelos, args):
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    dir_out = os.path.join(args.salida, grupo, nombre)
    os.makedirs(os.path.join(dir_out, "capturas"), exist_ok=True)

    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        print(f"[!] No se pudo abrir {ruta}")
        return None
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    n_usar = n_total // args.stride if n_total else 0
    if args.max_frames:
        n_usar = min(n_usar or args.max_frames, args.max_frames)
    idx_capturas = set(np.linspace(0, max(n_usar - 1, 0), args.capturas).astype(int).tolist()) if n_usar else set()

    writer = None
    filas = []
    i_leido = i_proc = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i_leido += 1
        if (i_leido - 1) % args.stride:
            continue
        if args.max_frames and i_proc >= args.max_frames:
            break
        if args.recorte_sup > 0:   # opcional: descartar cielo / capó
            frame = frame[int(frame.shape[0] * args.recorte_sup):]
        if args.recorte_inf > 0:
            frame = frame[:frame.shape[0] - int(frame.shape[0] * args.recorte_inf)]
        h, w = frame.shape[:2]
        y_eval = h * args.fila_eval

        paneles = []
        for m in modelos:
            out = m.inferir(frame)
            xl, xr = m.bordes_carril(out, y_eval)
            fila = {"grupo": grupo, "video": nombre, "frame": i_leido - 1,
                    "modelo": m.nombre, "t_modelo_ms": out["t_modelo_ms"],
                    "t_total_ms": out["t_total_ms"],
                    "x_izq": xl, "x_der": xr,
                    "borde_izq": xl is not None, "borde_der": xr is not None}
            if xl is not None and xr is not None:
                fila["centro_norm"] = ((xl + xr) / 2 - w / 2) / w     # 0 = centrado, <0 = carril a la izquierda
                fila["ancho_norm"] = (xr - xl) / w
            fila.update(m.cobertura(out))
            filas.append(fila)

            vis = marcar_fila(m.dibujar(frame, out), y_eval, xl, xr)
            esc = args.ancho_panel / w
            vis = cv2.resize(vis, (args.ancho_panel, int(h * esc)))
            paneles.append(rotulo(vis, f"{m.nombre} ({m.variante})",
                                  f"modelo: {out['t_modelo_ms']:.1f} ms   total: {out['t_total_ms']:.1f} ms"))

        combo = np.hstack(paneles)
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(dir_out, "comparacion.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_src / args.stride, (combo.shape[1], combo.shape[0]))
        writer.write(combo)
        if i_proc in idx_capturas:
            cv2.imwrite(os.path.join(dir_out, "capturas", f"frame_{i_leido - 1:06d}.png"), combo)
        i_proc += 1
        if i_proc % 50 == 0:
            print(f"   {nombre}: {i_proc}/{n_usar or '?'} cuadros", flush=True)
    cap.release()
    if writer:
        writer.release()
    if not filas:
        return None

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(dir_out, "frames.csv"), index=False)
    graficar_centro(df, os.path.join(dir_out, "centro_carril.png"), nombre)
    return df


def graficar_centro(df, ruta, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for mod, g in df.groupby("modelo"):
        if "centro_norm" in g:
            ax.plot(g["frame"], g["centro_norm"], label=mod, lw=1.4)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("cuadro")
    ax.set_ylabel("centro del carril\n(desvío / ancho imagen)")
    ax.set_title(f"Centro de carril estimado - {titulo}")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def resumir(df, warmup):
    res = []
    for (grupo, video, mod), g in df.groupby(["grupo", "video", "modelo"], sort=False):
        g = g.sort_values("frame")
        t = g["t_modelo_ms"].iloc[warmup:] if len(g) > warmup else g["t_modelo_ms"]
        tt = g["t_total_ms"].iloc[warmup:] if len(g) > warmup else g["t_total_ms"]
        c = g["centro_norm"] if "centro_norm" in g else pd.Series(dtype=float)
        salto = c.diff().abs().dropna()
        fila = {
            "grupo": grupo, "video": video, "modelo": mod, "cuadros": len(g),
            "lat_modelo_ms_mediana": t.median(), "lat_modelo_ms_p95": t.quantile(.95),
            "fps_modelo": 1000 / t.median(), "fps_total": 1000 / tt.median(),
            "pct_ambos_bordes": 100 * (g["borde_izq"] & g["borde_der"]).mean(),
            "pct_algun_borde": 100 * (g["borde_izq"] | g["borde_der"]).mean(),
            "jitter_centro_mediana": salto.median() if len(salto) else np.nan,
            "pct_saltos_grandes": 100 * (salto > 0.05).mean() if len(salto) else np.nan,
            "ancho_carril_std": g["ancho_norm"].std() if "ancho_norm" in g else np.nan,
        }
        for extra in ("pct_area_manejable", "pct_lineas", "n_carriles"):
            if extra in g:
                fila[extra + "_media"] = g[extra].mean()
        res.append(fila)
    res = pd.DataFrame(res)
    # Acuerdo entre modelos: diferencia del centro estimado cuando ambos lo detectan
    if "centro_norm" in df:
        p = df.pivot_table(index=["grupo", "video", "frame"], columns="modelo", values="centro_norm")
        if p.shape[1] == 2:
            d = (p.iloc[:, 0] - p.iloc[:, 1]).abs().groupby(level=["grupo", "video"]).median()
            res["desacuerdo_centro_mediana"] = res.apply(lambda r: d.get((r.grupo, r.video), np.nan), axis=1)
    return res


def graficar_resumen(res, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res.groupby(["grupo", "modelo"]).agg(
        fps=("fps_modelo", "median"), ambos=("pct_ambos_bordes", "mean"),
        jitter=("jitter_centro_mediana", "median")).reset_index()
    grupos = list(g["grupo"].unique())
    modelos = list(g["modelo"].unique())
    colores = {"TwinLiteNetPlus": "#2a6fdb", "UFLDv2": "#e07b24"}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, tit in zip(axs, ["fps", "ambos", "jitter"],
                            ["Velocidad (FPS, solo red)", "Cuadros con ambos bordes (%)",
                             "Jitter del centro (menor = más estable)"]):
        x = np.arange(len(grupos))
        wbar = 0.8 / len(modelos)
        for j, mod in enumerate(modelos):
            v = [g[(g.grupo == gr) & (g.modelo == mod)][col].mean() for gr in grupos]
            ax.bar(x + j * wbar - 0.4 + wbar / 2, v, wbar, label=mod, color=colores.get(mod))
        ax.set_xticks(x, grupos)
        ax.set_title(tit, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    aqui = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--videos", nargs="+", required=True, help="archivos o carpetas de video (cada carpeta = un grupo, ej. unlam / bfmc)")
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--tl-repo", default=os.path.join(aqui, "repos", "TwinLiteNetPlus"))
    ap.add_argument("--tl-config", default="large", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--tl-weight", default=None, help="ej. pesos/twinlite/large.pth")
    ap.add_argument("--tl-img-size", type=int, default=640)
    ap.add_argument("--ufld-repo", default=os.path.join(aqui, "repos", "ultra-fast-lane-detection-v2"))
    ap.add_argument("--ufld-config", default="culane_res18", help="nombre del config (culane_res18, culane_res34, tusimple_res18, tusimple_res34) o ruta .py")
    ap.add_argument("--ufld-weight", default=None, help="ej. pesos/ufld/culane_res18.pth")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--half", action="store_true", help="FP16 para TwinLiteNetPlus en GPU")
    ap.add_argument("--stride", type=int, default=1, help="procesar 1 de cada N cuadros")
    ap.add_argument("--max-frames", type=int, default=0, help="límite de cuadros por video (0 = todos)")
    ap.add_argument("--fila-eval", type=float, default=0.85, help="altura relativa (0-1) de la fila donde se miden los bordes del carril")
    ap.add_argument("--recorte-sup", type=float, default=0.0, help="fracción superior a descartar (cielo)")
    ap.add_argument("--recorte-inf", type=float, default=0.0, help="fracción inferior a descartar (capó)")
    ap.add_argument("--ancho-panel", type=int, default=640)
    ap.add_argument("--capturas", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=10, help="cuadros iniciales excluidos del cálculo de velocidad")
    ap.add_argument("--sin-pesos", action="store_true", help="solo para probar el pipeline con pesos aleatorios")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.sin_pesos and (not args.tl_weight or not args.ufld_weight):
        ap.error("Faltan --tl-weight y/o --ufld-weight (o usá --sin-pesos para una prueba)")
    ucfg = args.ufld_config
    if not ucfg.endswith(".py"):
        ucfg = os.path.join(args.ufld_repo, "configs", ucfg + ".py")

    print(f"Dispositivo: {args.device}")
    modelos = [
        TwinLite(args.tl_repo, args.tl_config, None if args.sin_pesos else args.tl_weight,
                 args.device, args.tl_img_size, args.half),
        UFLD(args.ufld_repo, ucfg, None if args.sin_pesos else args.ufld_weight, args.device),
    ]
    for m in modelos:
        print(f"  {m.nombre:16s} {m.variante:14s} {m.params / 1e6:7.2f} M parámetros")

    videos = listar_videos(args.videos)
    if not videos:
        sys.exit("No encontré videos en " + " ".join(args.videos))
    os.makedirs(args.salida, exist_ok=True)

    dfs = []
    for grupo, ruta in videos:
        print(f"-> [{grupo}] {ruta}")
        df = procesar_video(grupo, ruta, modelos, args)
        if df is not None:
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    res = resumir(df, args.warmup)
    res.to_csv(os.path.join(args.salida, "resumen.csv"), index=False)
    with open(os.path.join(args.salida, "resumen.md"), "w") as f:
        f.write(f"Dispositivo: `{args.device}` ({torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'})\n\n")
        for m in modelos:
            f.write(f"- **{m.nombre}** ({m.variante}): {m.params / 1e6:.2f} M parámetros\n")
        f.write("\n" + res.round(3).to_markdown(index=False) + "\n")
    graficar_resumen(res, os.path.join(args.salida, "graficos_resumen.png"))
    with open(os.path.join(args.salida, "config_corrida.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("\n" + res.round(2).to_string(index=False))
    print(f"\nListo. Resultados en {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
