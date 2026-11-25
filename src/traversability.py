"""Clase de terreno -> nivel de transitabilidad, resumen por superficie y superposición."""
import cv2
import numpy as np

from src.config import CLASS_NAMES as _CFG_NAMES, COLOR_MAP as _CFG_COLORS

LEVELS = ["transitable", "precaución", "no transitable"]
LEVEL_COLORS = {
    "transitable": (76, 175, 80),
    "precaución": (233, 185, 73),
    "no transitable": (184, 74, 51),
}

# Mapeo por ID de clase de RUGD.
#  - Tierra, grava y piedra: el robot puede transitar (definido por el equipo).
#  - Árboles y arbustos: no puede pasar (definido por el equipo).
#  - Pasto: NO está definido en el requisito; se asigna a "precaución" como supuesto.
#    Para tratarlo como no transitable, cambia el nivel del ID 3.
TRAVERSABILITY = {
    1: "transitable",       # dirt   (tierra)
    11: "transitable",      # gravel (grava)
    21: "transitable",      # rock   (piedra)
    3: "precaución",        # grass  (pasto)  <- supuesto, ajustable
    4: "no transitable",    # tree   (árbol)
    19: "no transitable",   # bush   (arbusto)
}
DEFAULT_LEVEL = "precaución"  # clases sin mapeo: se trata con cautela

ES_NAMES = {1: "tierra", 3: "pasto", 4: "árbol", 11: "grava", 19: "arbusto", 21: "piedra"}
_DEFAULT_NAMES = {1: "dirt", 3: "grass", 4: "tree", 11: "gravel", 19: "bush", 21: "rock"}
_DEFAULT_COLORS = {
    1: (108, 64, 20), 3: (0, 102, 0), 4: (0, 255, 0),
    11: (255, 128, 0), 19: (255, 153, 204), 21: (153, 204, 255),
}
# Nombres y colores por ID (config del usuario tiene prioridad; añade la grava si falta)
NAMES = {**_DEFAULT_NAMES, **_CFG_NAMES}
COLORS = {**_DEFAULT_COLORS, **_CFG_COLORS}


def level_map(ids):
    """Matriz (H, W) de IDs de clase -> índice de nivel en LEVELS."""
    out = np.full(ids.shape, LEVELS.index(DEFAULT_LEVEL), dtype=np.uint8)
    for cid, level in TRAVERSABILITY.items():
        out[ids == cid] = LEVELS.index(level)
    return out


def render_overlay(img, ids, mode="transitabilidad", alpha=0.5):
    """Superpone sobre la imagen un mapa de colores por clase o por nivel."""
    color = np.zeros(img.shape, dtype=np.uint8)
    if mode == "transitabilidad":
        lv = level_map(ids)
        for i, level in enumerate(LEVELS):
            color[lv == i] = LEVEL_COLORS[level]
    else:
        for cid in np.unique(ids):
            color[ids == cid] = COLORS.get(int(cid), (128, 128, 128))
    return cv2.addWeighted(img, 1 - alpha, color, alpha, 0)


def summarize(ids, conf, names=None):
    """Porcentaje de cada superficie y de cada nivel, y confianza media.

    Returns (filas_por_superficie, {nivel: %}, confianza_media_global)
    """
    names = names or ES_NAMES
    total = ids.size
    rows = []
    for cid in np.unique(ids):
        m = ids == cid
        rows.append({
            "Superficie": names.get(int(cid), str(int(cid))),
            "% de píxeles": 100.0 * m.sum() / total,
            "Confianza media": float(conf[m].mean()),
            "Nivel": TRAVERSABILITY.get(int(cid), DEFAULT_LEVEL),
        })
    rows.sort(key=lambda r: -r["% de píxeles"])
    lv = level_map(ids)
    levels = {LEVELS[i]: 100.0 * float((lv == i).mean()) for i in range(len(LEVELS))}
    return rows, levels, float(conf.mean())
