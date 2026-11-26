"""Configuración central: clases objetivo de RUGD y colores de visualización."""

# IDs de clase de RUGD usados en el proyecto
TARGET_CLASSES = [1, 3, 4, 19, 21]

CLASS_NAMES = {
    1: "dirt",
    3: "grass",
    4: "tree",
    19: "bush",
    21: "rock",
}

# Color RGB con el que se pinta cada clase en la máscara predicha
COLOR_MAP = {
    1: (108, 64, 20),      # dirt
    3: (0, 102, 0),        # grass
    4: (0, 255, 0),        # tree
    19: (255, 153, 204),   # bush
    21: (153, 204, 255),   # rock
}

DEFAULT_PATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Rutas por defecto (opcional)
# Si las defines aquí puedes ejecutar `python train.py` (o F5 en el editor)
# sin argumentos. Los argumentos de línea de comandos siempre tienen prioridad.
# ---------------------------------------------------------------------------
# Usa r"..." para que Windows no interprete las barras invertidas.
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent


def _first_existing(*paths):
    for p in paths:
        if p and _Path(p).exists():
            return str(p)
    return str(paths[0]) if paths else None


DEFAULT_IMAGES_DIR = _first_existing(
    r"C:\Users\Brayan\Documents\Maestria\ML\RUGD_frames-with-annotations\trail",
    _ROOT / "data" / "RUGD_frames-with-annotations",
)
DEFAULT_MASKS_DIR = _first_existing(
    r"C:\Users\Brayan\Documents\Maestria\ML\RUGD_annotations",
    _ROOT / "data" / "RUGD_annotations",
)
DEFAULT_CLASS_DICT = _first_existing(
    _ROOT / "data" / "RUGD_annotations" / "RUGD_annotation-colormap.txt",
    _ROOT / "dict (1).txt",
)

DEFAULT_TEST_SCENES = "trail-5,trail-12,park-2"
