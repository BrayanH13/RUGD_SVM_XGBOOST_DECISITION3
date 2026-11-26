"""Carga de RUGD: diccionario de clases, máscaras, parches y construcción del dataset."""
from collections import Counter, defaultdict
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


def _color_keys(class_dict):
    items = sorted(((r << 16) | (g << 8) | b, idx) for (r, g, b), idx in class_dict.items())
    keys = np.array([k for k, _ in items], dtype=np.int64)
    ids = np.array([i for _, i in items], dtype=np.int64)
    return keys, ids


def mask_to_ids(mask_path, class_dict):
    """Convierte una máscara RGB en una matriz (H, W) de IDs de clase."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask is None:
        mask = np.array(Image.open(mask_path).convert("RGB"))
    else:
        mask = mask[:, :, ::-1]
    code = ((mask[:, :, 0].astype(np.int64) << 16) | (mask[:, :, 1].astype(np.int64) << 8)
            | mask[:, :, 2].astype(np.int64))
    keys, ids = _color_keys(class_dict)
    pos = np.clip(np.searchsorted(keys, code), 0, len(keys) - 1)
    return np.where(keys[pos] == code, ids[pos], 0)


def enumerate_patches(mask_ids, patch_size=64, target_classes=None, stats=None):
    """Divide la máscara en parches sin traslape y devuelve (y, x, etiqueta) de los aceptados.

    La etiqueta de cada parche es la clase dominante (moda) de la máscara. Los parches
    incompletos en los bordes se descartan. Si ``target_classes`` está definido, solo se
    aceptan los parches cuya etiqueta esté en esa lista.
    """
    stats = stats if stats is not None else Counter()
    h, w = mask_ids.shape
    nh, nw = h // patch_size, w // patch_size
    if nh == 0 or nw == 0:
        return []
    m = mask_ids[:nh * patch_size, :nw * patch_size]
    blocks = m.reshape(nh, patch_size, nw, patch_size).transpose(0, 2, 1, 3).reshape(nh * nw, -1)
    k = int(blocks.max()) + 1
    offs = (np.arange(nh * nw) * k)[:, None]
    counts = np.bincount((blocks + offs).ravel(), minlength=nh * nw * k).reshape(nh * nw, k)
    labels = counts.argmax(1)

    stats["candidatos"] += len(labels)
    for lab, c in zip(*np.unique(labels, return_counts=True)):
        stats[f"dominante_{int(lab)}"] += int(c)
    keep = np.ones(len(labels), bool) if target_classes is None else np.isin(labels, list(target_classes))
    stats["fuera_de_clases"] += int((~keep).sum())
    idx = np.nonzero(keep)[0]
    return [(int(i // nw) * patch_size, int(i % nw) * patch_size, int(labels[i])) for i in idx]


def generate_patches(img, mask_ids, patch_size=64, target_classes=None):
    """Devuelve (parches, etiquetas) de una imagen. Ver ``enumerate_patches``."""
    coords = enumerate_patches(mask_ids, patch_size, target_classes)
    patches = [img[y:y + patch_size, x:x + patch_size] for y, x, _ in coords]
    return patches, [label for _, _, label in coords]


IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_SUFFIXES = ("_orig", "_mask", "_masks", "_gt", "_label", "_labels", "_annotation",
             "_annotations", "_color", "_seg", "_segmentation")


def _list_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)


def _norm(stem):
    """Nombre sin extensión, en minúsculas y sin sufijos típicos de máscara (_mask, _gt...)."""
    s = stem.lower()
    for suf in _SUFFIXES:
        if s.endswith(suf):
            return s[:-len(suf)]
    return s


def pair_images_and_masks(images_dir, masks_dir, scene_glob=None, frame_step=1):
    """Empareja cada imagen con su máscara.

    Orden de intentos:
      1. Mismo nombre sin importar la extensión (trail_00001.png <-> trail_00001.jpg).
      2. Mismo nombre ignorando sufijos de máscara (trail_00001 <-> trail_00001_mask).
      3. Si nada coincide y hay igual número de imágenes que de máscaras, se emparejan
         por orden alfabético (con aviso).

    Parameters
    ----------
    images_dir : escena (``trail``) o carpeta padre con varias (``trail``, ``trail-3``...).
    masks_dir : carpeta de máscaras (plana o con subcarpetas).
    scene_glob : subcarpetas de ``images_dir`` a usar, p. ej. ``"trail*"``. Si ninguna
        coincide, ``images_dir`` se trata como una sola escena.
    frame_step : usa 1 de cada N imágenes de cada escena.

    Returns
    -------
    pairs : lista de (ruta_imagen, ruta_máscara)
    summary : {escena: (n_imágenes_usadas, n_con_máscara)}
    """
    images_dir = Path(images_dir)
    scene_dirs = []
    if scene_glob:
        scene_dirs = sorted(p for p in images_dir.glob(scene_glob) if p.is_dir())
    if not scene_dirs:
        scene_dirs = [images_dir]

    all_masks = _list_images(masks_dir)
    by_stem, by_norm = defaultdict(list), defaultdict(list)
    for m in all_masks:
        by_stem[m.stem.lower()].append(m)
        by_norm[_norm(m.stem)].append(m)

    step = max(1, frame_step)
    pairs, summary, all_imgs = [], {}, []
    for scene in scene_dirs:
        imgs = _list_images(scene)
        all_imgs += imgs
        used = imgs[::step]
        found = []
        for p in used:
            cands = by_stem.get(p.stem.lower()) or by_norm.get(_norm(p.stem))
            if cands:
                same_scene = [c for c in cands if c.parent.name == scene.name]
                found.append((p, (same_scene or cands)[0]))
        summary[scene.name] = (len(used), len(found))
        pairs.extend(found)

    if not pairs and all_imgs and len(all_imgs) == len(all_masks):
        pairs = list(zip(sorted(all_imgs), sorted(all_masks)))[::step]
        summary = {"(emparejado por orden alfabético: los nombres no coinciden)": (len(pairs), len(pairs))}
    return pairs, summary


def diagnose_pairing(images_dir, masks_dir, scene_glob=None):
    """Imprime qué ve el programa en las carpetas y qué nombres no se pudieron emparejar."""
    for label, folder in (("imágenes", images_dir), ("máscaras", masks_dir)):
        p = Path(folder)
        if not p.exists():
            print(f"  {label}: la carpeta NO existe -> {folder}")
            return
        print(f"  {label}: {len(_list_images(p))} archivos en {folder}")

    images_dir = Path(images_dir)
    scenes = []
    if scene_glob:
        scenes = [s for s in sorted(images_dir.glob(scene_glob)) if s.is_dir()]
        print(f"  subcarpetas que coinciden con '{scene_glob}': "
              + (", ".join(s.name for s in scenes) if scenes else "ninguna (se usa images-dir como una sola escena)"))
    all_imgs = [f for s in (scenes or [images_dir]) for f in _list_images(s)]
    all_masks = _list_images(masks_dir)

    pairs, _ = pair_images_and_masks(images_dir, masks_dir, scene_glob, 1)
    used_i = {p[0] for p in pairs}
    used_m = {p[1] for p in pairs}
    no_mask = [f.name for f in all_imgs if f not in used_i]
    no_img = [f.name for f in all_masks if f not in used_m]
    print(f"  imágenes SIN máscara: {len(no_mask)} | ejemplos: {', '.join(no_mask[:4]) or '-'}")
    print(f"  máscaras SIN imagen:  {len(no_img)} | ejemplos: {', '.join(no_img[:4]) or '-'}")
    if len(all_masks) < len(all_imgs):
        print(f"  -> Hay menos máscaras ({len(all_masks)}) que imágenes ({len(all_imgs)}): "
              "probablemente --masks-dir no apunta a la carpeta que tiene TODAS las anotaciones.")


def read_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_pair(args):
    img_path, mask_path, class_dict, patch_size, target_classes = args
    stats = Counter()
    mask_ids = mask_to_ids(mask_path, class_dict)
    coords = enumerate_patches(mask_ids, patch_size, target_classes, stats)
    if not coords:
        return None, [], stats
    img = read_rgb(img_path)
    patches = np.stack([img[y:y + patch_size, x:x + patch_size] for y, x, _ in coords])
    return patches, [lab for _, _, lab in coords], stats


def build_dataset(pairs, class_dict, feature_fn, patch_size=64, target_classes=None, stats=None,
                  workers=8, return_patches=False, gpu_batch=8192, progress=True, return_groups=False):
    """Recorre los pares (imagen, máscara) y devuelve X, y y la lista de parches originales.

    ``stats`` (Counter opcional) se rellena con los conteos de parches candidatos y aceptados.
    """
    from concurrent.futures import ThreadPoolExecutor
    from src.gpu_features import batch_features

    stats = stats if stats is not None else Counter()
    X, y_out, patches_list, groups = [], [], [], []
    buf, buf_n = [], 0

    def flush():
        nonlocal buf, buf_n
        if buf:
            X.append(batch_features(feature_fn, np.concatenate(buf)))
            buf, buf_n = [], 0

    jobs = ((ip, mp, class_dict, patch_size, target_classes) for ip, mp in pairs)
    total = len(pairs)
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for i, ((img_path, _), (patches, labels, st)) in enumerate(zip(pairs, ex.map(_load_pair, jobs)), 1):
            stats.update(st)
            if patches is not None:
                buf.append(patches)
                buf_n += len(patches)
                y_out.extend(labels)
                groups.extend([Path(img_path).parent.name] * len(labels))
                if return_patches:
                    patches_list.extend(patches)
                if buf_n >= gpu_batch:
                    flush()
            if progress and (i % 50 == 0 or i == total):
                print(f"\r  imágenes procesadas: {i}/{total} | parches: {len(y_out)}", end="", flush=True)
    flush()
    if progress and total:
        print()
    stats["aceptados"] = len(y_out)
    X = np.concatenate(X) if X else np.zeros((0, 0), np.float32)
    if return_groups:
        return X, np.array(y_out), patches_list, np.array(groups)
    return X, np.array(y_out), patches_list
