import argparse
import csv
import time
from pathlib import Path

import joblib
import numpy as np

from src.dataset import read_rgb
from src.gpu_features import batch_features, describe_device, make_feature_fn, resolve_device
from src.gpu_models import prepare_for_inference
from src.inference import image_to_patches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models-dir", default="outputs")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"Dispositivo: {describe_device(device)}")
    files = sorted(Path(args.images_dir).rglob("*.png"))
    step = max(1, len(files) // args.n_images)
    imgs = [read_rgb(f) for f in files[::step][:args.n_images]]
    print(f"Imágenes: {len(imgs)} de {imgs[0].shape[1]}x{imgs[0].shape[0]} px")

    rows = []
    for path in sorted(Path(args.models_dir).glob("model_*.joblib")):
        b = joblib.load(path)
        clf = prepare_for_inference(b["model"], device.type)
        fn = make_feature_fn(b["feature_params"], device=device)
        ps = b["patch_size"]
        p0, _ = image_to_patches(imgs[0], ps)
        clf.predict(batch_features(fn, p0))
        t_feat, t_pred = [], []
        for img in imgs:
            patches, _ = image_to_patches(img, ps)
            t0 = time.perf_counter()
            X = batch_features(fn, patches)
            t1 = time.perf_counter()
            clf.predict(X)
            t2 = time.perf_counter()
            t_feat.append(t1 - t0)
            t_pred.append(t2 - t1)
        mb = path.stat().st_size / 2**20
        tf, tp = 1000 * np.mean(t_feat), 1000 * np.mean(t_pred)
        row = {"modelo": path.stem.replace("model_", ""), "tamano_MB": round(mb, 2),
               "caracteristicas_ms": round(tf, 1), "prediccion_ms": round(tp, 1),
               "total_ms": round(tf + tp, 1), "imagenes_por_s": round(1000 / (tf + tp), 1),
               "parches_por_imagen": len(patches)}
        rows.append(row)
        print(row)

    out = Path(args.models_dir) / "medicion_modelos.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
