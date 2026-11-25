"""Entrena clasificadores (SVM, XGBoost, Árbol de decisión) sobre parches de RUGD.

Por defecto entrena los tres, imprime el reporte de cada uno y muestra:
  - una matriz de confusión por modelo (lado a lado)
  - la predicción de cada modelo sobre una imagen

Ejemplos:
    python train.py                       # usa las rutas de src/config.py, entrena los 3 modelos
    python train.py --model svm           # solo un modelo
    python train.py --scene-glob "trail*" --frame-step 5 --no-show
"""
import argparse
import random
from collections import Counter
from functools import partial
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (DEFAULT_CLASS_DICT, DEFAULT_IMAGES_DIR, DEFAULT_MASKS_DIR,
                        DEFAULT_PATCH_SIZE, TARGET_CLASSES)
from src.traversability import COLORS as COLOR_MAP, NAMES as CLASS_NAMES
from src.dataset import (build_dataset, diagnose_pairing, load_class_dict,
                        pair_images_and_masks, read_rgb)
from src.features import extract_features

MODEL_KEYS = ["svm", "xgboost", "tree"]
DISPLAY_NAMES = {"svm": "SVM", "xgboost": "XGBoost", "tree": "Árbol de decisión"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Carpeta con los frames RGB (.png)")
    p.add_argument("--masks-dir", default=DEFAULT_MASKS_DIR, help="Carpeta con las anotaciones RGB (.png)")
    p.add_argument("--class-dict", default=DEFAULT_CLASS_DICT, help="Archivo de colormap de RUGD")
    p.add_argument("--scene-glob", default="trail*",
                   help="Subcarpetas de --images-dir a usar, p. ej. 'trail*'. Usa '*' para todas.")
    p.add_argument("--frame-step", type=int, default=1,
                   help="Usar 1 de cada N imágenes por escena (acelera el entrenamiento)")
    p.add_argument("--classes", default=None,
                   help="IDs RUGD a entrenar, separados por coma. Ej. 1,3,4,11,19,21 (incluye grava=11).\n"
                        "Por defecto usa TARGET_CLASSES de src/config.py")
    p.add_argument("--model", choices=MODEL_KEYS + ["all"], default="all",
                   help="Modelo a entrenar; 'all' entrena y compara los tres (default)")
    p.add_argument("--max-depth", type=int, default=None,
                   help="Profundidad máxima (solo tree; None = sin límite)")
    p.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    p.add_argument("--lbp-mode", choices=["raw", "hist"], default="raw",
                   help="raw: 100 primeros valores LBP (original) | hist: histograma de 256 bins")
    p.add_argument("--color-bins", type=int, default=8,
                   help="Bins por canal del histograma de color (8 = original; más = más fino)")
    p.add_argument("--color-mode", choices=["joint", "channels"], default="joint",
                   help="joint: histograma 3D (bins^3 valores) | channels: uno por canal (3*bins valores)")
    p.add_argument("--hog-orientations", type=int, default=9,
                   help="Bins de orientación de HOG (9 = original)")
    p.add_argument("--hog-cell", type=int, default=8,
                   help="Tamaño de celda de HOG en px (8 = original; menor = más detalle y más dimensiones)")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--no-show", action="store_true",
                   help="No abrir ventanas (solo guardar las figuras en --output-dir)")
    args = p.parse_args()
    missing = [n for n, v in [("--images-dir", args.images_dir), ("--masks-dir", args.masks_dir),
                              ("--class-dict", args.class_dict)] if not v]
    if missing:
        p.error("faltan las rutas " + ", ".join(missing) + ". Pásalas por argumento o defínelas "
                "en src/config.py (DEFAULT_IMAGES_DIR, DEFAULT_MASKS_DIR, DEFAULT_CLASS_DICT).")
    return args


def build_model(name, seed, max_depth=None):
    if name == "svm":
        return SVC(kernel="linear")
    if name == "tree":
        return DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    from xgboost import XGBClassifier  # import perezoso: solo si se usa
    return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                         eval_metric="mlogloss", random_state=seed)


def main():
    args = parse_args()
    show = not args.no_show
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.visualize import plot_confusion_matrices, plot_predictions, reconstruct_image

    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_params = {
        "lbp_mode": args.lbp_mode,
        "color_bins": args.color_bins,
        "color_mode": args.color_mode,
        "hog_orientations": args.hog_orientations,
        "hog_cell": args.hog_cell,
    }
    feature_fn = partial(extract_features, **feature_params)
    class_dict = load_class_dict(args.class_dict)

    # 1. Dataset
    pairs, summary = pair_images_and_masks(
        args.images_dir, args.masks_dir, args.scene_glob, args.frame_step
    )
    print(f"Escenas encontradas: {len(summary)}")
    for scene, (n_img, n_ok) in summary.items():
        flag = "" if n_img == n_ok else f"  <-- {n_img - n_ok} sin máscara"
        print(f"  {scene:<12} imágenes: {n_img:<5} con máscara: {n_ok}{flag}")
    n_missing = sum(n_img - n_ok for n_img, n_ok in summary.values())
    if pairs and n_missing:
        print(f"AVISO: {n_missing} imágenes no tienen máscara y NO se usan en el entrenamiento. Detalle:")
        diagnose_pairing(args.images_dir, args.masks_dir, args.scene_glob)
    if not pairs:
        print("No se encontraron pares imagen/máscara. Esto es lo que se ve en las carpetas:")
        diagnose_pairing(args.images_dir, args.masks_dir, args.scene_glob)
        raise SystemExit("Revisa que --images-dir y --masks-dir apunten a las carpetas correctas y que los nombres coincidan.")
    print(f"Pares imagen/máscara totales: {len(pairs)}")
    stats = Counter()
    classes = [int(c) for c in args.classes.split(",")] if args.classes else TARGET_CLASSES
    X, y, _ = build_dataset(pairs, class_dict, feature_fn, args.patch_size, classes, stats=stats)
    print(f"Parches candidatos: {stats['candidatos']} | fuera de clases objetivo: {stats['fuera_de_clases']}"
          f" | aceptados: {stats['aceptados']}")
    dom = {int(k.split("_")[1]): v for k, v in stats.items() if k.startswith("dominante_")}
    print("Clase dominante de los parches candidatos (ID: cantidad):", dict(sorted(dom.items())))
    if X.size == 0:
        msg = "Ningún parche cumple los filtros."
        if set(dom) <= {0}:
            msg += (" Todos los parches salen como clase 0 (sin etiqueta): los colores de tus máscaras no "
                    "coinciden con --class-dict. Diagnostica con: python check_masks.py --masks-dir "
                    "<carpeta de máscaras> --class-dict <dict.txt>")
        elif not (set(dom) & set(classes)):
            msg += f" Las clases presentes {sorted(dom)} no incluyen ninguna de las objetivo {classes}. Revisa --classes."
        else:
            msg += " Revisa --patch-size y los datos."
        raise SystemExit(msg)
    print(f"Parches: {X.shape[0]} | Dimensión del vector: {X.shape[1]}")

    # 2. Etiquetas a índices continuos [0..N-1] (XGBoost lo exige)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    names = [CLASS_NAMES.get(int(c), str(c)) for c in le.classes_]
    counts = np.bincount(y_enc)
    print("Parches por clase:", {n: int(c) for n, c in zip(names, counts)})

    # 3. Split (el mismo para todos los modelos, estratificado si es posible)
    stratify = y_enc if counts.min() >= 2 else None
    if stratify is None:
        print("Aviso: hay una clase con 1 parche; el split no se estratifica.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    # 4. Entrenar y evaluar cada modelo
    keys = MODEL_KEYS if args.model == "all" else [args.model]
    results, trained = {}, {}
    for key in keys:
        label = DISPLAY_NAMES[key]
        print(f"\n===== {label} =====")
        clf = build_model(key, args.seed, args.max_depth)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, labels=np.arange(len(names)),
                                    target_names=names, zero_division=0))
        results[label] = (y_test, y_pred)
        trained[label] = clf
        joblib.dump({"model": clf, "label_encoder": le, "patch_size": args.patch_size,
                     "lbp_mode": args.lbp_mode,
                     "feature_params": feature_params}, out_dir / f"model_{key}.joblib")

    # 5. Resumen comparativo
    print("\n===== Resumen (conjunto de test) =====")
    print(f"{'Modelo':<20}{'Accuracy':>10}{'F1 macro':>11}")
    for label, (yt, yp) in results.items():
        print(f"{label:<20}{accuracy_score(yt, yp):>10.3f}"
              f"{f1_score(yt, yp, average='macro', zero_division=0):>11.3f}")

    # 6. Figuras: matriz de confusión por modelo + predicción por modelo
    plot_confusion_matrices(results, names, out_dir / "confusion_matrices.png", close=not show)

    img_path, _ = random.choice(pairs)
    img = read_rgb(img_path)
    masks = {label: reconstruct_image(img, clf, le, feature_fn, COLOR_MAP, args.patch_size)
             for label, clf in trained.items()}
    plot_predictions(img, masks, {int(c): COLOR_MAP[int(c)] for c in le.classes_ if int(c) in COLOR_MAP}, CLASS_NAMES, out_dir / "predictions.png", close=not show)

    print(f"\nResultados guardados en: {out_dir.resolve()}")
    if show:
        plt.show()  # abre las dos figuras y espera a que las cierres


if __name__ == "__main__":
    main()