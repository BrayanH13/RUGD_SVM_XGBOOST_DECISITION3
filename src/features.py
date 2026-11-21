"""Extracción de características por parche: HOG + LBP + histograma de color."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def extract_features(patch, lbp_mode="raw"):
    """Convierte un parche RGB (H, W, 3, uint8) en un vector 1D de características.

    Vector final = [ HOG | LBP | histograma de color 8x8x8 ]

    Parameters
    ----------
    patch : np.ndarray
        Parche RGB.
    lbp_mode : {"raw", "hist"}
        "raw"  -> comportamiento original del proyecto: primeros 100 valores
                  de la imagen LBP aplanada (NO es un histograma).
        "hist" -> histograma normalizado de 256 bins de los códigos LBP
                  (invariante a la posición; recomendado).
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # HOG: gradientes orientados, celdas 8x8, bloques 2x2, 9 orientaciones
    hog_feat = hog(
        gray,
        pixels_per_cell=(8, 8),
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

    # Histograma de color 3D: 8 bins por canal -> 512 valores
    color_hist = cv2.calcHist(
        [patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    ).flatten()

    return np.concatenate([hog_feat, lbp_feat, color_hist])
