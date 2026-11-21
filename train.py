"""Entrena un clasificador (SVM o XGBoost) sobre parches de RUGD.

Ejemplo:
    python train.py --images-dir data/RUGD_frames-with-annotations/trail \
                    --masks-dir  data/RUGD_annotations/trail \
                    --class-dict data/RUGD_annotation-colormap.txt \
                    --model svm --patch-size 64
"""
import argparse
import random
from functools import partial
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src.config import CLASS_NAMES, COLOR_MAP, DEFAULT_PATCH_SIZE, TARGET_CLASSES
from src.dataset import build_dataset, load_class_dict, pair_images_and_masks, read_rgb
from src.features import extract_features
from src.visualize import reconstruct_image, save_comparison, save_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images-dir", required=True, help="Carpeta con los frames RGB (.png)")
    p.add_argument("--masks-dir", required=True, help="Carpeta con las anotaciones RGB (.png)")
    p.add_argument("--class-dict", required=True, help="Archivo de colormap de RUGD")
    p.add_argument("--model", choices=["svm", "xgboost"], default="svm")
    p.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    p.add_argument("--lbp-mode", choices=["raw", "hist"], default="raw",
                   help="raw: 100 primeros valores LBP (original) | hist: histograma de 256 bins")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--show", action="store_true", help="Mostrar la figura final en pantalla")
    return p.parse_args()


def build_model(name, seed):
    if name == "svm":
        return SVC(kernel="linear")
    from xgboost import XGBClassifier  # import perezoso: solo si se usa
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=seed,
    )


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_fn = partial(extract_features, lbp_mode=args.lbp_mode)
    class_dict = load_class_dict(args.class_dict)

    # 1. Dataset
    pairs = pair_images_and_masks(args.images_dir, args.masks_dir)
    if not pairs:
        raise SystemExit("No se encontraron pares imagen/máscara con el mismo nombre.")
    print(f"Pares imagen/máscara: {len(pairs)}")
    X, y, _ = build_dataset(pairs, class_dict, feature_fn, args.patch_size, TARGET_CLASSES)
    print(f"Parches: {X.shape[0]} | Dimensión del vector: {X.shape[1]}")

    # 2. Etiquetas a índices continuos (XGBoost lo exige; SVM no lo necesita pero es inocuo)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    names = [CLASS_NAMES.get(int(c), str(c)) for c in le.classes_]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=args.test_size, random_state=args.seed
    )

    # 4. Entrenamiento y evaluación
    clf = build_model(args.model, args.seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=names))
    save_confusion_matrix(y_test, y_pred, names, out_dir / f"confusion_{args.model}.png")

    # 5. Guardar modelo
    joblib.dump(
        {"model": clf, "label_encoder": le, "patch_size": args.patch_size,
         "lbp_mode": args.lbp_mode},
        out_dir / f"model_{args.model}.joblib",
    )

    # 6. Reconstrucción visual sobre una imagen aleatoria
    img_path, _ = random.choice(pairs)
    img = read_rgb(img_path)
    pred_mask = reconstruct_image(img, clf, le, feature_fn, COLOR_MAP, args.patch_size)
    save_comparison(img, pred_mask, f"Predicción {args.model.upper()}",
                    out_dir / f"prediction_{args.model}.png", show=args.show)
    print(f"Resultados guardados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
