"""Carga de RUGD: diccionario de clases, máscaras, parches y construcción del dataset."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_class_dict(path_txt):
    """Lee el archivo de colormap de RUGD.

    Formato por línea: ``idx nombre r g b``.  Devuelve {(r, g, b): idx}.
    """
    class_dict = {}
    with open(path_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                idx = int(parts[0])
                r, g, b = map(int, parts[2:])
                class_dict[(r, g, b)] = idx
    return class_dict


def mask_to_ids(mask_path, class_dict):
    """Convierte una máscara RGB en una matriz (H, W) de IDs de clase."""
    mask = np.array(Image.open(mask_path).convert("RGB"))
    h, w, _ = mask.shape
    ids = np.zeros((h, w), dtype=np.int64)
    for (r, g, b), idx in class_dict.items():
        ids[(mask[:, :, 0] == r) & (mask[:, :, 1] == g) & (mask[:, :, 2] == b)] = idx
    return ids


def generate_patches(img, mask_ids, patch_size=64, target_classes=None):
    """Divide la imagen en parches sin solapamiento.

    La etiqueta de cada parche es la clase dominante (moda) de su máscara.
    Los parches incompletos en los bordes se descartan.
    """
    h, w, _ = img.shape
    patches, labels = [], []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img[y:y + patch_size, x:x + patch_size]
            mask_patch = mask_ids[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                label = np.bincount(mask_patch.flatten()).argmax()
                if target_classes is None or label in target_classes:
                    patches.append(patch)
                    labels.append(label)
    return patches, labels


def pair_images_and_masks(images_dir, masks_dir):
    """Empareja imágenes y máscaras por nombre de archivo (no por orden).

    Evita desalineaciones cuando las carpetas tienen distinto número de archivos.
    """
    masks = {p.name: p for p in Path(masks_dir).rglob("*.png")}
    pairs = []
    for img_path in sorted(Path(images_dir).rglob("*.png")):
        if img_path.name in masks:
            pairs.append((img_path, masks[img_path.name]))
    return pairs


def read_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_dataset(pairs, class_dict, feature_fn, patch_size=64, target_classes=None):
    """Recorre los pares (imagen, máscara) y devuelve X, y, parches originales."""
    X, y, patches_list = [], [], []
    for img_path, mask_path in pairs:
        img = read_rgb(img_path)
        mask_ids = mask_to_ids(mask_path, class_dict)
        patches, labels = generate_patches(img, mask_ids, patch_size, target_classes)
        for patch, label in zip(patches, labels):
            X.append(feature_fn(patch))
            y.append(label)
            patches_list.append(patch)
    return np.array(X), np.array(y), patches_list
