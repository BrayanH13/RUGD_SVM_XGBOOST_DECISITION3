"""Usa un modelo entrenado (.joblib) para segmentar una imagen nueva.

Ejemplo:
    python predict.py --model outputs/model_svm.joblib --image foto.png --output pred.png
"""
import argparse
from functools import partial

import joblib
import matplotlib
matplotlib.use("Agg")

from src.config import COLOR_MAP
from src.dataset import read_rgb
from src.features import extract_features
from src.visualize import reconstruct_image, save_comparison


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True, help="Archivo .joblib generado por train.py")
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="prediction.png")
    args = p.parse_args()

    bundle = joblib.load(args.model)
    # modelos antiguos solo guardaban lbp_mode; los nuevos guardan todos los parámetros
    params = bundle.get("feature_params") or {"lbp_mode": bundle.get("lbp_mode", "raw")}
    feature_fn = partial(extract_features, **params)
    img = read_rgb(args.image)
    mask = reconstruct_image(img, bundle["model"], bundle["label_encoder"],
                             feature_fn, COLOR_MAP, bundle["patch_size"])
    save_comparison(img, mask, "Predicción", args.output)
    print(f"Guardado en {args.output}")


if __name__ == "__main__":
    main()
