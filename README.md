# Clasificación de Landmarks Naturales en RUGD

Clasificación de terreno y vegetación en entornos no estructurados usando el dataset
**RUGD** con características clásicas de visión por computador (**HOG + LBP + histograma de color**)
y clasificadores **SVM** y **XGBoost**, sin redes neuronales.

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
                          SVM (lineal) / XGBoost
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
  --model svm            # o: xgboost
```

Opciones principales:

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--model` | `svm` | `svm` (kernel lineal) o `xgboost` |
| `--patch-size` | `64` | Tamaño del parche en píxeles |
| `--lbp-mode` | `raw` | `raw` (100 primeros valores, versión original) o `hist` (histograma 256 bins) |
| `--test-size` | `0.3` | Fracción para test |
| `--output-dir` | `outputs` | Carpeta de resultados |
| `--show` | — | Mostrar la figura final en pantalla |

Genera en `outputs/`: reporte de clasificación (consola), matriz de confusión, imagen de predicción
y el modelo `model_<svm|xgboost>.joblib`.

### Predecir sobre una imagen nueva

```bash
python predict.py --model outputs/model_svm.joblib --image foto.png --output pred.png
```

## Estructura del repositorio

```
.
├── src/
│   ├── config.py       # clases objetivo y mapa de colores
│   ├── dataset.py      # lectura de máscaras, parches, emparejado, construcción de X/y
│   ├── features.py     # HOG + LBP + histograma de color
│   └── visualize.py    # reconstrucción de máscara, superposición, matriz de confusión
├── train.py            # entrenamiento y evaluación (SVM / XGBoost)
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
- **Escalado.** El SVM lineal es sensible a la escala de las características (HOG ∈ [0,1], histograma de
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
