# Clasificación de Landmarks Naturales en RUGD

Clasificación de terreno y vegetación en entornos no estructurados usando el dataset
**RUGD** con características clásicas de visión por computador (**HOG + LBP + histograma de color**)
y clasificadores **SVM**, **XGBoost** y **Árbol de decisión**, sin redes neuronales.

> Proyecto de la Maestría en Inteligencia Artificial — Reconocimiento de Patrones.

## Objetivo

Predecir, para cada parche de una imagen, una de las siguientes clases de RUGD:

| ID RUGD | Clase | Color en la máscara |
|:-------:|-------|:-------------------:|
| 1  | dirt  | marrón (108, 64, 20) |
| 3  | grass | verde oscuro (0, 102, 0) |
| 4  | tree  | verde brillante (0, 255, 0) |
| 19 | bush  | rosa (255, 153, 204) |
| 21 | rock  | azul claro (153, 204, 255) |

## Dataset

[RUGD — Robot Unstructured Ground Driving Dataset](http://rugd.vision) (Wigness et al., IROS 2019):
imágenes RGB de robots terrestres en entornos naturales, con anotación semántica por píxel.
El dataset **no se incluye** en este repositorio; descárgalo del sitio oficial.

Estructura esperada (ejemplo):

```
data/
├── RUGD_frames-with-annotations/trail/*.png     # imágenes
├── RUGD_annotations/trail/*.png                 # máscaras RGB
└── RUGD_annotation-colormap.txt                 # "idx nombre r g b" por línea
```

> Imágenes y máscaras se emparejan **por nombre de archivo**, no por orden alfabético.

## Pipeline

```
Imagen RGB ──► parches 64×64 ──► etiqueta = clase dominante de la máscara
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                  ▼
                   HOG               LBP          Histograma de color
                    └─────────────────┼──────────────────┘
                                      ▼
                         np.concatenate → vector 1D
                                      ▼
                          SVM (lineal) / XGBoost / Árbol de decisión
                                      ▼
                    clase por parche → máscara reconstruida
```

### Características por parche

| Descriptor | Qué captura | Parámetros | Dim. (parche 64) | Dim. (parche 32) |
|-----------|-------------|-----------|:----------------:|:----------------:|
| **HOG** | Forma / bordes (gradientes orientados) | celdas 8×8, bloques 2×2, 9 orient. | 1764 | 324 |
| **LBP** | Textura local | P=8, R=1 | 100 | 100 |
| **Hist. color** | Cromaticidad | 8×8×8 bins RGB | 512 | 512 |
| **Total** | | | **2376** | **936** |

Con `--lbp-mode hist` el bloque LBP pasa a ser un histograma de 256 bins (ver notas abajo).

**HOG.** Calcula el gradiente de cada píxel (magnitud y ángulo), agrupa los ángulos en histogramas
de 9 bins por celda de 8×8 px y normaliza por bloques de 2×2 celdas. Un parche de 64×64 produce
8×8 celdas y 7×7 bloques de 36 valores: 7·7·36 = 1764.

**LBP.** Para cada píxel compara sus 8 vecinos con el valor central (≥ centro → 1, < centro → 0) y
lee los 8 bits en círculo para obtener un código entre 0 y 255. Es invariante a cambios monótonos de
iluminación porque solo compara vecinos entre sí.

**Histograma de color.** Histograma 3D de 8 bins por canal (512 combinaciones), normalmente muy
informativo para separar tierra, pasto y roca.

## Procesamiento de datos: parches

La imagen se divide en parches cuadrados sin traslape de `--patch-size` píxeles (64 por defecto). Cada parche recibe la etiqueta de la clase dominante de su máscara, y solo se conservan los de las clases objetivo. `train.py` imprime cuántos parches fueron candidatos y cuántos aceptados, y la clase dominante encontrada.

## Ajustar la resolución de las características

Los bins controlan qué tan fino es cada descriptor. Dimensión del vector con parches de 64 px:

| Configuración | Dimensión |
|---------------|:---------:|
| Original (`--color-bins 8`, 9 orientaciones, celda 8, LBP raw) | 2376 |
| `--color-bins 16` (joint) | 5960 |
| `--color-bins 32 --color-mode channels` | 1960 |
| `--hog-orientations 18` | 4140 |
| `--hog-cell 4` | 8712 |
| `--color-bins 16 --hog-orientations 18 --lbp-mode hist` | 7880 |

Más bins no garantiza mejor resultado: aumenta las dimensiones, hace los histogramas más dispersos
(con 4096 bins un parche de 64×64 = 4096 píxeles deja la mayoría casi vacíos) y alarga el entrenamiento.
Conviene cambiar un parámetro a la vez y comparar con la tabla resumen que imprime `train.py`.
`--color-mode channels` permite bins muy finos sin que el vector se dispare.

## Instalación

```bash
git clone https://github.com/<tu-usuario>/rugd-landmarks-classification.git
cd rugd-landmarks-classification
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

### Entrenar y evaluar

```bash
python train.py \
  --images-dir data/RUGD_frames-with-annotations/trail \
  --masks-dir  data/RUGD_annotations/trail \
  --class-dict data/RUGD_annotation-colormap.txt \
  --model all            # o: svm | xgboost | tree
```

Opciones principales:

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--classes` | `src/config.py` | IDs RUGD a entrenar, separados por coma. Ej. `1,3,4,11,19,21` incluye la grava (11) |
| `--model` | `all` | `all` entrena y compara los tres; o `svm`, `xgboost`, `tree` por separado |
| `--max-depth` | `None` | Profundidad máxima del árbol (solo `--model tree`); limitarla reduce el sobreajuste |
| `--scene-glob` | `trail*` | Subcarpetas de `--images-dir` a usar (`trail`, `trail-3`, ...). `*` = todas |
| `--frame-step` | `1` | Usa 1 de cada N imágenes por escena (acelera el entrenamiento) |
| `--patch-size` | `64` | Tamaño del parche en píxeles |
| `--lbp-mode` | `raw` | `raw` (100 primeros valores, versión original) o `hist` (histograma 256 bins) |
| `--color-bins` | `8` | Bins por canal del histograma de color (más = más fino) |
| `--color-mode` | `joint` | `joint` (histograma 3D, bins³ valores) o `channels` (uno por canal, 3·bins valores) |
| `--hog-orientations` | `9` | Bins de orientación de HOG |
| `--hog-cell` | `8` | Tamaño de celda de HOG en píxeles (menor = más detalle espacial) |
| `--test-size` | `0.3` | Fracción para test |
| `--output-dir` | `outputs` | Carpeta de resultados |
| `--no-show` | — | No abrir ventanas; solo guardar las figuras (útil en servidores o scripts) |

Por defecto entrena los tres modelos con el mismo split y muestra dos ventanas: una **matriz de confusión
por modelo** (lado a lado) y la **predicción de cada modelo** sobre una imagen, con leyenda de colores.
También imprime el reporte de clasificación de cada uno y una tabla resumen (accuracy y F1 macro).

Genera en `outputs/`: `confusion_matrices.png`, `predictions.png` y un `model_<svm|xgboost|tree>.joblib`
por modelo.

**¿Prefieres no escribir argumentos?** Define tus rutas una sola vez en `src/config.py`
(`DEFAULT_IMAGES_DIR`, `DEFAULT_MASKS_DIR`, `DEFAULT_CLASS_DICT`) y ejecuta `python train.py`
o presiona F5 en tu editor. Los argumentos de línea de comandos tienen prioridad sobre esos valores.

### Predecir sobre una imagen nueva

```bash
python predict.py --model outputs/model_svm.joblib --image foto.png --output pred.png
```

## Aplicación web: transitabilidad

La aplicación permite subir una **imagen o un video** del recorrido y devuelve la imagen con un mapa de
colores superpuesto (por clase de terreno o por nivel de transitabilidad), el **porcentaje de cada tipo de
superficie** y la **confianza** de la predicción. En video analiza 1 de cada N fotogramas y grafica la
transitabilidad a lo largo del recorrido.

```bash
# 1. Entrenar incluyendo la grava (clase 11); genera outputs/model_*.joblib
python train.py --classes 1,3,4,11,19,21
# 2. Lanzar la aplicación desde la carpeta del proyecto
streamlit run app.py
```

Mapeo de clase a nivel (definido en `src/traversability.py`):

| Nivel | Superficies |
|-------|-------------|
| Transitable | tierra (1), grava (11), piedra (21) |
| Precaución | pasto (3) — **supuesto**: el requisito no lo define |
| No transitable | árbol (4), arbusto (19) |

Para tratar el pasto como no transitable, cambia su nivel en `TRAVERSABILITY`. Las clases sin mapeo se
tratan como precaución. La grava solo se detecta si el modelo se entrenó con `--classes` incluyéndola.

Notas:
- **Confianza.** XGBoost y el árbol entregan probabilidades. El SVM lineal no: se usa el softmax de su margen,
  que es un puntaje relativo y no una probabilidad calibrada (la app lo avisa).
- **Alcance.** Es una estimación visual por parches; no sustituye a un sistema de seguridad del robot.

## Estructura del repositorio

```
.
├── src/
│   ├── config.py       # clases objetivo y mapa de colores
│   ├── dataset.py      # lectura de máscaras, parches, emparejado, construcción de X/y
│   ├── features.py     # HOG + LBP + histograma de color
│   ├── inference.py    # predicción por imagen/video con confianza por píxel
│   ├── traversability.py  # clase -> nivel de transitabilidad, resumen, superposición
│   └── visualize.py    # reconstrucción de máscara, superposición, matriz de confusión
├── app.py              # aplicación web (Streamlit)
├── train.py            # entrenamiento y evaluación (SVM / XGBoost / árbol)
├── predict.py          # inferencia con un modelo guardado
├── requirements.txt
└── LICENSE
```

## Notas técnicas

- **XGBoost y las etiquetas.** XGBoost exige clases consecutivas `0..N-1`, pero RUGD usa IDs dispersos
  (`1, 3, 4, 19, 21`). Se usa `LabelEncoder` y se invierte con `inverse_transform` al colorear.
  El mismo encoder se aplica al SVM para mantener un solo flujo.
- **Bordes de la imagen.** Los parches incompletos se descartan al entrenar, pero al reconstruir la
  imagen se rellena (`reflect`) hasta un múltiplo del tamaño de parche para no dejar píxeles sin predecir.
- **Etiqueta por parche.** Se asigna la clase dominante; un parche que mezcla clases (p. ej. borde
  sendero/arbusto) queda etiquetado con una sola, lo que introduce ruido inherente.
- **LBP `raw` vs `hist`.** La versión original toma `lbp_img.ravel()[:100]`: los primeros 100 *píxeles*
  de la imagen LBP (la primera fila y parte de la segunda), **no** un histograma. Es sensible a la
  posición y descarta casi toda la textura del parche. El modo `hist` calcula el histograma de códigos
  sobre todo el parche y es la variante recomendada para comparar resultados.
- **Árbol de decisión.** Sin límite de profundidad (`--max-depth` por defecto) memoriza el entrenamiento y
  tiende a sobreajustar; prueba valores como 5–15. No necesita escalado de características.
- **Escalado.** El SVM lineal (no el árbol ni XGBoost) es sensible a la escala de las características (HOG ∈ [0,1], histograma de
  color en conteos). Probar `StandardScaler` suele mejorar y acelerar la convergencia.
- **Partición.** El `train_test_split` es aleatorio por parche, por lo que parches de una misma imagen
  pueden caer en train y test; para una evaluación más estricta conviene separar por imagen o por escena.

## Resultados

Ejemplo de predicción SVM (parches 64×64) — izquierda: imagen original, derecha: máscara predicha.

<!-- Reemplaza con tu imagen: ![Resultado SVM](docs/resultado_svm.png) -->

| Modelo | Accuracy | F1 macro |
|--------|:--------:|:--------:|
| SVM lineal | _por completar_ | _por completar_ |
| XGBoost | _por completar_ | _por completar_ |
| Árbol de decisión | _por completar_ | _por completar_ |

## Trabajo futuro

- Escalado de características (`StandardScaler`) y búsqueda de hiperparámetros (`GridSearchCV` / Optuna).
- LBP uniforme / multirradio y comparación `raw` vs `hist`.
- Métricas por píxel (IoU por clase) en lugar de solo por parche.
- Validación por escena/imagen para evitar fuga de información entre train y test.
- Comparación con CNN / Vision Transformers.

## Referencia

Wigness, M., Eum, S., Rogers, J. G., Han, D., Kwon, H. *A RUGD Dataset for Autonomous Navigation and
Visual Perception in Unstructured Outdoor Environments.* IROS, 2019.

## Licencia

MIT — ver [LICENSE](LICENSE).
