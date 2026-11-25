"""Diagnostica si los colores de las máscaras coinciden con el diccionario de clases.

Ejemplo:
    python check_masks.py --masks-dir data/annotations --class-dict data/dict.txt
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_dict(path):
    out = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 5:
                out[(int(p[2]), int(p[3]), int(p[4]))] = (int(p[0]), p[1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--class-dict", required=True)
    ap.add_argument("--n", type=int, default=3, help="Cuántas máscaras revisar")
    args = ap.parse_args()

    cd = read_dict(args.class_dict)
    files = sorted(Path(args.masks_dir).rglob("*.png"))[:args.n]
    if not files:
        raise SystemExit(f"No hay PNG en {args.masks_dir}")
    print(f"Diccionario: {len(cd)} colores | máscaras a revisar: {len(files)}\n")

    for f in files:
        im = Image.open(f)
        rgb = np.array(im.convert("RGB")).reshape(-1, 3)
        colors, counts = np.unique(rgb, axis=0, return_counts=True)
        total = counts.sum()
        print(f"== {f.name}  (modo {im.mode}, {im.size[0]}x{im.size[1]}, {len(colors)} colores distintos)")
        unmatched = 0.0
        for i in np.argsort(-counts)[:12]:
            c = tuple(int(v) for v in colors[i])
            hit = cd.get(c)
            print(f"   {str(c):<18} {100 * counts[i] / total:5.1f} %  ->  "
                  + (f"{hit[1]} (ID {hit[0]})" if hit else "SIN COINCIDENCIA"))
        for c, n in zip(colors, counts):
            if tuple(int(v) for v in c) not in cd:
                unmatched += n / total * 100
        print(f"   Píxeles sin coincidencia en el diccionario: {unmatched:.1f} %")
        if (colors[:, 0] == colors[:, 1]).all() and (colors[:, 1] == colors[:, 2]).all() and len(colors) <= 40:
            print("   AVISO: todos los colores son grises. Parecen máscaras de IDs (0, 1, 2...), no de colores.")
        elif unmatched > 5:
            print("   AVISO: hay muchos colores que no están en el dict.txt; usa el RUGD_annotation-colormap.txt oficial.")
        print()


if __name__ == "__main__":
    main()
