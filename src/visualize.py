"""Reconstrucción de la máscara predicha y figuras de resultados."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.metrics import ConfusionMatrixDisplay


def reconstruct_image(img, clf, le, feature_fn, color_map, patch_size=64):
    """Clasifica la imagen por parches y devuelve una máscara de color (H, W, 3).

    - Rellena (reflect) los bordes para que ningún píxel quede sin predecir.
    - Predice todos los parches en una sola llamada (más rápido).
    - Usa ``le.inverse_transform`` para volver de índices [0..N-1] a IDs RUGD.
    """
    h, w, _ = img.shape
    pad_h = (-h) % patch_size
    pad_w = (-w) % patch_size
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W, _ = padded.shape

    coords, feats = [], []
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            coords.append((y, x))
            feats.append(feature_fn(padded[y:y + patch_size, x:x + patch_size]))

    pred_ids = le.inverse_transform(clf.predict(np.array(feats)))

    out = np.zeros((H, W, 3), dtype=np.uint8)
    for (y, x), cid in zip(coords, pred_ids):
        out[y:y + patch_size, x:x + patch_size] = color_map.get(int(cid), (0, 0, 0))
    return out[:h, :w]


def overlay(img, mask, alpha=0.45):
    """Superpone la máscara de color sobre la imagen original."""
    return cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)


def save_comparison(img, pred_mask, title, path, show=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Imagen original")
    axes[1].imshow(pred_mask)
    axes[1].set_title(title)
    axes[2].imshow(overlay(img, pred_mask))
    axes[2].set_title("Superposición")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, display_labels, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=np.arange(len(display_labels)), display_labels=display_labels,
        normalize="true", cmap="Greens", ax=ax, values_format=".2f",
    )
    ax.set_title("Matriz de confusión (normalizada)")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results, class_names, path, close=True):
    """Una matriz de confusión por modelo, lado a lado, en una sola figura.

    results : {nombre_modelo: (y_true, y_pred)}  (índices codificados 0..N-1)
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.4), squeeze=False)
    for ax, (name, (y_true, y_pred)) in zip(axes[0], results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=np.arange(len(class_names)),
            display_labels=class_names, normalize="true", cmap="Greens",
            ax=ax, values_format=".2f", colorbar=False,
        )
        ax.set_title(f"{name}\nAccuracy: {(np.asarray(y_true) == np.asarray(y_pred)).mean():.3f}")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Matrices de confusión (normalizadas por fila)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    if close:
        plt.close(fig)
    return fig


def plot_predictions(img, masks, color_map, class_names, path, close=True):
    """Imagen original + la máscara predicha por cada modelo, con leyenda de colores.

    masks : {nombre_modelo: máscara (H, W, 3)}
    class_names : {id_rugd: nombre}
    """
    n = 1 + len(masks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.8), squeeze=False)
    axes[0][0].imshow(img)
    axes[0][0].set_title("Imagen original")
    for ax, (name, mask) in zip(axes[0][1:], masks.items()):
        ax.imshow(mask)
        ax.set_title(f"Predicción {name}")
    for ax in axes[0]:
        ax.axis("off")
    handles = [Patch(facecolor=tuple(c / 255 for c in color_map[k]), edgecolor="gray",
                     label=class_names.get(k, str(k))) for k in color_map]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    if close:
        plt.close(fig)
    return fig
