"""Inferencia sobre imágenes y videos completos: mapa de clase y confianza por píxel."""
import cv2
import numpy as np


def resize_max_width(img, max_width):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    return cv2.resize(img, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)


def class_scores(clf, X, n_classes):
    """Puntajes por clase (n, n_classes) y si son probabilidades reales.

    - Modelos con predict_proba (XGBoost, árbol, SVM con probability=True): probabilidades.
    - SVM sin probabilidades: softmax del margen (puntaje relativo, NO calibrado).
    """
    scores, is_prob = None, False
    try:
        scores = clf.predict_proba(X)
        is_prob = True
    except (AttributeError, NotImplementedError):
        pass
    if scores is None and hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        if s.ndim == 1:
            s = np.c_[-s, s]
        e = np.exp(s - s.max(axis=1, keepdims=True))
        scores = e / e.sum(axis=1, keepdims=True)
    if scores is None:
        scores = np.eye(n_classes)[clf.predict(X).astype(int)]

    full = np.zeros((len(X), n_classes), dtype=np.float32)
    cls = np.asarray(getattr(clf, "classes_", np.arange(scores.shape[1]))).astype(int)
    if scores.shape[1] == len(cls):
        full[:, cls] = scores
    else:
        full[:, :scores.shape[1]] = scores
    return full, is_prob


def predict_map(img, clf, le, feature_fn, patch_size=64):
    """Clasifica la imagen por parches de ``patch_size`` y devuelve (ids, confianza, es_probabilidad).

    Los bordes se rellenan (reflect) hasta un múltiplo del tamaño de parche para que
    ningún píxel quede sin predecir.
      ids  : (H, W) IDs de clase de RUGD
      conf : (H, W) puntaje de la clase elegida en [0, 1]
    """
    h, w = img.shape[:2]
    Hp = max(patch_size, -(-h // patch_size) * patch_size)
    Wp = max(patch_size, -(-w // patch_size) * patch_size)
    padded = np.pad(img, ((0, Hp - h), (0, Wp - w), (0, 0)), mode="reflect")

    coords, feats = [], []
    for y in range(0, Hp, patch_size):
        for x in range(0, Wp, patch_size):
            coords.append((y, x))
            feats.append(feature_fn(padded[y:y + patch_size, x:x + patch_size]))

    scores, is_prob = class_scores(clf, np.array(feats), len(le.classes_))
    full = np.zeros((Hp, Wp, scores.shape[1]), dtype=np.float32)
    for (y, x), sc in zip(coords, scores):
        full[y:y + patch_size, x:x + patch_size] = sc
    full = full[:h, :w]
    return np.asarray(le.classes_)[full.argmax(axis=2)], full.max(axis=2), is_prob


def sample_video_frames(path, step=30, max_frames=20, max_width=640):
    """Toma 1 de cada `step` fotogramas (hasta `max_frames`). Devuelve (frames_rgb, fps, total)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames, i = [], 0
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(resize_max_width(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), max_width))
            i += 1
    finally:
        cap.release()
    return frames, fps, total
