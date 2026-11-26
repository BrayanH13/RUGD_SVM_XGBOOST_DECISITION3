"""Usa un modelo entrenado (.joblib) para segmentar una imagen nueva.

Ejemplo:
    python predict.py --model outputs/model_svm.joblib --image foto.png --output pred.png
"""
import argparse

import joblib
import matplotlib
matplotlib.use("Agg")

from src.config import COLOR_MAP
from src.dataset import read_rgb
from src.gpu_features import describe_device, make_feature_fn, resolve_device
from src.gpu_models import prepare_for_inference
from src.visualize import reconstruct_image, save_comparison


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True, help="Archivo .joblib generado por train.py")
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="prediction.png")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    bundle = joblib.load(args.model)
    # modelos antiguos solo guardaban lbp_mode; los nuevos guardan todos los parámetros
    params = bundle.get("feature_params") or {"lbp_mode": bundle.get("lbp_mode", "raw")}
    device = resolve_device(args.device)
    print(f"Dispositivo: {describe_device(device)}")
    feature_fn = make_feature_fn(params, device=device)
    prepare_for_inference(bundle["model"], device.type)
    img = read_rgb(args.image)
    mask = reconstruct_image(img, bundle["model"], bundle["label_encoder"],
                             feature_fn, COLOR_MAP, bundle["patch_size"])
    save_comparison(img, mask, "Predicción", args.output)
    print(f"Guardado en {args.output}")


if __name__ == "__main__":
    main()
