import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import DEFAULT_CLASS_DICT, DEFAULT_IMAGES_DIR, DEFAULT_MASKS_DIR
from src.dataset import enumerate_patches, mask_to_ids, read_rgb

TARGET = [1, 3, 4, 11, 19, 21]
ES = {1: "tierra", 3: "pasto", 4: "árbol", 11: "grava", 19: "arbusto", 21: "roca"}


def read_dict(path):
    names, colors = {}, {}
    for line in open(path):
        p = line.split()
        if len(p) == 5:
            i = int(p[0])
            names[i], colors[i] = p[1], tuple(int(v) / 255 for v in p[2:])
    return names, colors


def patch_label_image(ids, colors, patch=64):
    h, w = ids.shape
    out = np.zeros((h, w, 3))
    for y, x, lab in enumerate_patches(ids, patch):
        out[y:y + patch, x:x + patch] = colors.get(lab, (0, 0, 0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    ap.add_argument("--masks-dir", default=DEFAULT_MASKS_DIR)
    ap.add_argument("--class-dict", default=DEFAULT_CLASS_DICT)
    ap.add_argument("--frame-step", type=int, default=10, help="Máscaras analizadas: 1 de cada N")
    ap.add_argument("--example", default=None, help="Nombre de un frame para la figura de ejemplo")
    ap.add_argument("--output-dir", default="outputs_eda")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names, colors = read_dict(args.class_dict)
    class_dict = {tuple(int(round(c * 255)) for c in col): i for i, col in colors.items()}
    masks_root = Path(args.masks_dir)
    scenes = sorted(d for d in masks_root.iterdir() if d.is_dir())

    rows, total = [], Counter()
    for sc in scenes:
        files = sorted(sc.glob("*.png"))
        cnt = Counter()
        for f in files[::args.frame_step]:
            for _, _, lab in enumerate_patches(mask_to_ids(f, class_dict), 64):
                cnt[lab] += args.frame_step
        total.update(cnt)
        rows.append({"escena": sc.name, "frames": len(files),
                     **{ES[c]: cnt[c] for c in TARGET},
                     "otras": sum(v for k, v in cnt.items() if k not in TARGET)})
        print(rows[-1])
    with open(out / "eda_resumen_escenas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ks = [k for k, _ in total.most_common()]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([names.get(k, str(k)) for k in ks], [total[k] for k in ks],
           color=["tab:green" if k in TARGET else "tab:gray" for k in ks])
    ax.set_yscale("log")
    ax.set_ylabel("Parches 64x64 (estimado)")
    ax.set_title("Clase dominante por parche (verde: clases objetivo)")
    ax.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    fig.savefig(out / "eda_clases.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(rows))
    for c in TARGET:
        vals = np.array([r[ES[c]] for r in rows], float)
        tot = np.array([sum(r[ES[t]] for t in TARGET) for r in rows], float).clip(min=1)
        ax.bar([r["escena"] for r in rows], 100 * vals / tot, bottom=bottom, label=ES[c],
               color=colors[c], edgecolor="gray", linewidth=0.3)
        bottom += 100 * vals / tot
    ax.set_ylabel("% de parches (clases objetivo)")
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    fig.savefig(out / "eda_escenas.png", dpi=200)
    plt.close(fig)

    mask = (next(masks_root.rglob(args.example)) if args.example
            else sorted(scenes[0].glob("*.png"))[len(list(scenes[0].glob("*.png"))) // 2])
    img = next(Path(args.images_dir).rglob(mask.name))
    ids = mask_to_ids(mask, class_dict)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].imshow(read_rgb(img))
    axes[0].set_title("Imagen RGB")
    axes[1].imshow(read_rgb(mask))
    axes[1].set_title("Máscara de anotación")
    axes[2].imshow(patch_label_image(ids, colors))
    axes[2].set_title("Etiqueta por parche 64x64 (moda)")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    fig.savefig(out / "eda_ejemplo.png", dpi=200)
    plt.close(fig)
    print(f"Figuras en {out.resolve()}")


if __name__ == "__main__":
    main()
