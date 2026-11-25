"""Extracción de características por parche: HOG + LBP + histograma de color."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def extract_features(patch, lbp_mode="raw", color_bins=8, color_mode="joint",
                     hog_orientations=9, hog_cell=8):
    """Convierte un parche RGB (H, W, 3, uint8) en un vector 1D de características.

    Vector final = [ HOG | LBP | histograma de color ]

    Parameters
    ----------
    patch : np.ndarray
        Parche RGB.
    lbp_mode : {"raw", "hist"}
        "raw"  -> comportamiento original: primeros 100 valores de la imagen LBP
                  aplanada (NO es un histograma).
        "hist" -> histograma normalizado de 256 bins de los códigos LBP.
    color_bins : int
        Bins por canal del histograma de color. Más bins = color más fino.
    color_mode : {"joint", "channels"}
        "joint"    -> histograma 3D conjunto (color_bins ** 3 valores). Captura
                      combinaciones de color, pero crece muy rápido (16 -> 4096).
        "channels" -> un histograma por canal R, G, B (3 * color_bins valores).
                      Permite bins muy finos con pocas dimensiones.
    hog_orientations : int
        Bins de orientación de HOG (9 por defecto). Más bins = ángulo más fino.
    hog_cell : int
        Lado de la celda de HOG en píxeles (8 por defecto). Más pequeña = más
        detalle espacial y un vector mucho más largo.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # HOG: gradientes orientados
    hog_feat = hog(
        gray,
        orientations=hog_orientations,
        pixels_per_cell=(hog_cell, hog_cell),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    # LBP: 8 vecinos, radio 1 -> códigos en [0, 255]
    lbp_img = local_binary_pattern(gray, P=8, R=1)
    if lbp_mode == "raw":
        lbp_feat = lbp_img.ravel()[:100]
    elif lbp_mode == "hist":
        lbp_feat, _ = np.histogram(lbp_img, bins=256, range=(0, 256))
        lbp_feat = lbp_feat / max(lbp_feat.sum(), 1)
    else:
        raise ValueError(f"lbp_mode inválido: {lbp_mode!r} (usa 'raw' o 'hist')")

    # Histograma de color
    if color_mode == "joint":
        color_hist = cv2.calcHist(
            [patch], [0, 1, 2], None, [color_bins] * 3, [0, 256] * 3
        ).flatten()
    elif color_mode == "channels":
        color_hist = np.concatenate([
            cv2.calcHist([patch], [c], None, [color_bins], [0, 256]).flatten()
            for c in range(3)
        ])
    else:
        raise ValueError(f"color_mode inválido: {color_mode!r} (usa 'joint' o 'channels')")

    return np.concatenate([hog_feat, lbp_feat, color_hist])
