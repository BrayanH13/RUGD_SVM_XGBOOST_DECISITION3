import argparse
import csv
import itertools
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from src.config import (DEFAULT_CLASS_DICT, DEFAULT_IMAGES_DIR, DEFAULT_MASKS_DIR,
                        DEFAULT_TEST_SCENES)
from src.dataset import build_dataset, load_class_dict, pair_images_and_masks
from src.gpu_features import describe_device, make_feature_fn, resolve_device
from src.gpu_models import build_model
from src.traversability import NAMES as CLASS_NAMES

GRIDS = {
    "svm": {"svm_c": [0.01, 0.1, 1.0, 10.0]},
    "xgboost": {"max_depth": [4, 6, 8], "n_estimators": [100, 200], "learning_rate": [0.1]},
    "tree": {"max_depth": [5, 10, 15, 20]},
}
DISPLAY = {"svm": "SVM lineal", "xgboost": "XGBoost", "tree": "Árbol de decisión"}


def make_model(key, params, seed, device):
    if key == "svm":
        return build_model("svm", seed, device=device, svm_c=params["svm_c"], svm_iters=200)
    if key == "tree":
        return build_model("tree", seed, max_depth=params["max_depth"], device=device)
    clf = build_model("xgboost", seed, device=device)
    clf.set_params(**params)
    return clf


def stratified_sample(y, n, seed):
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    quota = {c: max(1, int(round(n * cnt / len(y)))) for c, cnt in zip(classes, counts)}
    for c, cnt in zip(classes, counts):
        quota[c] = max(quota[c], min(cnt, 2000))
    idx = [rng.choice(np.flatnonzero(y == c), min(quota[c], (y == c).sum()), replace=False)
           for c in classes]
    return np.sort(np.concatenate(idx))


def evaluate(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true), zero_division=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    ap.add_argument("--masks-dir", default=DEFAULT_MASKS_DIR)
    ap.add_argument("--class-dict", default=DEFAULT_CLASS_DICT)
    ap.add_argument("--scene-glob", default="*")
    ap.add_argument("--frame-step", type=int, default=4)
    ap.add_argument("--classes", default="1,3,4,11,19,21")
    ap.add_argument("--test-scenes", default=DEFAULT_TEST_SCENES)
    ap.add_argument("--val-scenes", default="trail-4,trail-6,park-1",
                    help="Escenas de validación para elegir hiperparámetros")
    ap.add_argument("--search-sample", type=int, default=30000,
                    help="Parches de train usados en la búsqueda (muestra estratificada)")
    ap.add_argument("--models", default="svm,xgboost,tree")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--gpu-batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="outputs_hparams")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Dispositivo: {describe_device(device)}")
    feature_params = dict(lbp_mode="raw", color_bins=8, color_mode="joint",
                          hog_orientations=9, hog_cell=8)
    feature_fn = make_feature_fn(feature_params, device=device, batch_size=args.gpu_batch)

    pairs, summary = pair_images_and_masks(args.images_dir, args.masks_dir, args.scene_glob,
                                           args.frame_step)
    print(f"Escenas: {len(summary)} | pares imagen/máscara: {len(pairs)}")
    classes = [int(c) for c in args.classes.split(",")]
    t0 = time.time()
    X, y, _, groups = build_dataset(pairs, load_class_dict(args.class_dict), feature_fn, 64,
                                    classes, stats=Counter(), workers=args.workers,
                                    gpu_batch=args.gpu_batch, return_groups=True)
    print(f"Características: {X.shape} en {time.time() - t0:.1f} s")
    le = LabelEncoder()
    y = le.fit_transform(y)
    names = [CLASS_NAMES.get(int(c), str(c)) for c in le.classes_]

    test_sc = [s.strip() for s in args.test_scenes.split(",")]
    val_sc = [s.strip() for s in args.val_scenes.split(",")]
    for s in test_sc + val_sc:
        if s not in set(groups):
            raise SystemExit(f"La escena {s!r} no está en los datos cargados")
    te = np.isin(groups, test_sc)
    va = np.isin(groups, val_sc)
    tr = ~(te | va)
    print(f"Train: {tr.sum()} | Validación ({', '.join(val_sc)}): {va.sum()} | "
          f"Test ({', '.join(test_sc)}): {te.sum()}")
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        print(f"  {nm:<5}", dict(zip(names, np.bincount(y[m], minlength=len(names)).tolist())))

    idx_search = np.flatnonzero(tr)[stratified_sample(y[tr], args.search_sample, args.seed)]
    Xs, ys = X[idx_search], y[idx_search]
    Xv, yv = X[va], y[va]
    print(f"Muestra para la búsqueda: {len(ys)} parches")

    rows, best = [], {}
    keys = [k.strip() for k in args.models.split(",")]
    for key in keys:
        grid = GRIDS[key]
        combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
        print(f"\n===== {DISPLAY[key]}: {len(combos)} combinaciones =====")
        for params in combos:
            t1 = time.time()
            clf = make_model(key, params, args.seed, device.type)
            if hasattr(clf, "verbose"):
                clf.verbose = False
            clf.fit(Xs, ys)
            acc, f1 = evaluate(yv, clf.predict(Xv))
            dt = time.time() - t1
            print(f"  {params}  ->  acc {acc:.3f} | F1 macro {f1:.3f} | {dt:.1f} s", flush=True)
            rows.append({"modelo": DISPLAY[key], **{f"param_{k}": v for k, v in params.items()},
                         "val_accuracy": round(acc, 4), "val_f1_macro": round(f1, 4),
                         "tiempo_s": round(dt, 1)})
            if key not in best or f1 > best[key][1]:
                best[key] = (params, f1)
        print(f"  MEJOR: {best[key][0]} (F1 macro val {best[key][1]:.3f})")

    fields = sorted({k for r in rows for k in r}, key=lambda k: (not k == "modelo", k))
    with open(out / "hparams_validacion.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        r = [x for x in rows if x["modelo"] == DISPLAY[key]]
        if key == "xgboost":
            for ne, lr in itertools.product(GRIDS[key]["n_estimators"], GRIDS[key]["learning_rate"]):
                s = [x for x in r if x["param_n_estimators"] == ne and x["param_learning_rate"] == lr]
                ax.plot([x["param_max_depth"] for x in s], [x["val_f1_macro"] for x in s], "o-",
                        label=f"{ne} árboles, lr={lr}")
            ax.set_xlabel("max_depth")
            ax.legend(fontsize=8)
        else:
            p = "param_svm_c" if key == "svm" else "param_max_depth"
            ax.plot([x[p] for x in r], [x["val_f1_macro"] for x in r], "o-")
            ax.set_xlabel("C" if key == "svm" else "max_depth")
            if key == "svm":
                ax.set_xscale("log")
        ax.set_title(DISPLAY[key])
        ax.set_ylabel("F1 macro (validación)")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "hparams.png", dpi=150)
    plt.close(fig)

    full = tr | va
    Xf, yf, Xt, yt = X[full], y[full], X[te], y[te]
    majority = np.bincount(yf).argmax()
    acc, f1 = evaluate(yt, np.full_like(yt, majority))
    final = [{"modelo": f"Línea base ({names[majority]})", "hiperparametros": "-",
              "test_accuracy": round(acc, 4), "test_f1_macro": round(f1, 4), "tiempo_s": ""}]
    print(f"\n===== Evaluación final en TEST (entrenado con {len(yf)} parches) =====")
    for key in keys:
        params = best[key][0]
        t1 = time.time()
        clf = make_model(key, params, args.seed, device.type)
        clf.fit(Xf, yf)
        dt = time.time() - t1
        yp = clf.predict(Xt)
        acc, f1 = evaluate(yt, yp)
        per = f1_score(yt, yp, average=None, labels=np.arange(len(names)), zero_division=0)
        print(f"\n{DISPLAY[key]} {params}: acc {acc:.3f} | F1 macro {f1:.3f} | {dt:.1f} s")
        print(classification_report(yt, yp, labels=np.arange(len(names)), target_names=names,
                                    zero_division=0))
        final.append({"modelo": DISPLAY[key], "hiperparametros": str(params),
                      "test_accuracy": round(acc, 4), "test_f1_macro": round(f1, 4),
                      "tiempo_s": round(dt, 1),
                      **{f"f1_{n}": round(v, 4) for n, v in zip(names, per)}})
        joblib.dump({"model": clf, "label_encoder": le, "patch_size": 64, "lbp_mode": "raw",
                     "feature_params": feature_params, "hiperparametros": params},
                    out / f"model_{key}.joblib")

    fields = list(dict.fromkeys(k for r in final for k in r))
    with open(out / "resultados_test.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(final)
    print("\n===== Resumen TEST =====")
    for r in final:
        print(f"{r['modelo']:<26} acc {r['test_accuracy']:.3f} | F1 macro {r['test_f1_macro']:.3f}"
              f" | {r['hiperparametros']}")
    print(f"\nResultados en {out.resolve()}")


if __name__ == "__main__":
    main()
