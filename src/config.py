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
# Usa r"..." para que Windows no interprete las barras invertidas.
# ---------------------------------------------------------------------------
DEFAULT_IMAGES_DIR = r"C:\Users\Brayan\Documents\Maestria\ML\RUGD_frames-with-annotations\trail"   # r"C:\ruta\RUGD_frames-with-annotations\trail"
DEFAULT_MASKS_DIR =r"C:\Users\Brayan\Documents\Maestria\ML\RUGD_annotations"     # r"C:\ruta\annotations"
DEFAULT_CLASS_DICT =  r"C:\Users\Brayan\Documents\Maestria\ML\rugd-landmarks-classification\dict (1).txt"
