"""Aplicación web: estimación de transitabilidad off-road.

Ejecutar desde la carpeta del proyecto:
    streamlit run app.py
Requiere un modelo entrenado con train.py (outputs/model_*.joblib).
"""
import os
import tempfile
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.features import extract_features
from src.inference import predict_map, resize_max_width, sample_video_frames
from src.traversability import (ES_NAMES, LEVEL_COLORS, LEVELS, TRAVERSABILITY,
                                render_overlay, summarize)

st.set_page_config(page_title="Transitabilidad off-road", layout="wide")
st.title("Estimación de transitabilidad off-road")
st.caption("Sube una imagen o un video del recorrido: el sistema marca la clase de terreno y su "
           "nivel de transitabilidad, con el porcentaje de cada superficie y la confianza.")


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


# ---------------------------- barra lateral ----------------------------
with st.sidebar:
    st.header("Modelo")
    found = sorted((Path(__file__).parent / "outputs").glob("*.joblib"))
    if found:
        model_path = str(st.selectbox("Modelo entrenado", found, format_func=lambda p: Path(p).name))
    else:
        model_path = st.text_input("Ruta del modelo (.joblib)", "outputs/model_xgboost.joblib")
    st.header("Predicción")
    overlay_mode = st.radio("Superposición", ["Transitabilidad", "Clase de terreno"])
    alpha = st.slider("Opacidad del mapa", 0.1, 0.9, 0.5, 0.05)
    max_w = st.select_slider("Ancho máximo de la imagen (px)", [480, 640, 800, 960], 640)
    st.header("Niveles de transitabilidad")
    for level in LEVELS:
        names = ", ".join(ES_NAMES.get(c, str(c)) for c, lv in TRAVERSABILITY.items() if lv == level)
        r, g, b = LEVEL_COLORS[level]
        st.markdown(f"<span style='color:rgb({r},{g},{b});font-weight:600'>■ {level}</span>: {names}",
                    unsafe_allow_html=True)

try:
    bundle = load_bundle(model_path)
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo cargar el modelo '{model_path}'. Entrena primero con train.py. Detalle: {e}")
    st.stop()

patch = bundle["patch_size"]
params = bundle.get("feature_params") or {"lbp_mode": bundle.get("lbp_mode", "raw")}
feature_fn = partial(extract_features, **params)
trained = [ES_NAMES.get(int(c), str(int(c))) for c in bundle["label_encoder"].classes_]
st.sidebar.caption("Clases del modelo: " + ", ".join(trained))
if not any(lv == "transitable" and c in bundle["label_encoder"].classes_
           for c, lv in TRAVERSABILITY.items()):
    st.warning("El modelo no tiene clases transitables.")


def analyze(rgb):
    return predict_map(rgb, bundle["model"], bundle["label_encoder"], feature_fn, patch)


def show_results(rgb, ids, conf, is_prob):
    mode = "transitabilidad" if overlay_mode.startswith("Trans") else "clase"
    c1, c2 = st.columns(2)
    c1.image(rgb, caption="Imagen original")
    c2.image(render_overlay(rgb, ids, mode, alpha), caption=f"Superposición: {overlay_mode}")
    rows, levels, mean_conf = summarize(ids, conf)
    cols = st.columns(4)
    for col, level in zip(cols, LEVELS):
        col.metric(level.capitalize(), f"{levels[level]:.1f} %")
    cols[3].metric("Confianza media", f"{mean_conf * 100:.0f} %")
    df = pd.DataFrame(rows)
    df["% de píxeles"] = df["% de píxeles"].round(1)
    df["Confianza media"] = (df["Confianza media"] * 100).round(0).astype(int).astype(str) + " %"
    st.dataframe(df, hide_index=True)
    if not is_prob:
        st.caption("La confianza es un puntaje relativo del SVM (softmax del margen), no una probabilidad "
                   "calibrada. XGBoost y el árbol entregan probabilidades.")


tab_img, tab_vid = st.tabs(["Imagen", "Video"])

# ------------------------------- imagen --------------------------------
with tab_img:
    up = st.file_uploader("Imagen del recorrido", type=["png", "jpg", "jpeg"], key="img")
    if up is not None and st.button("Analizar imagen"):
        rgb = resize_max_width(np.array(Image.open(up).convert("RGB")), max_w)
        with st.spinner("Analizando..."):
            ids, conf, is_prob = analyze(rgb)
        st.session_state["img_res"] = dict(name=up.name, rgb=rgb, ids=ids, conf=conf, is_prob=is_prob)
    res = st.session_state.get("img_res")
    if res:
        if up is not None and res["name"] != up.name:
            st.info("Hay una imagen nueva: presiona «Analizar imagen».")
        show_results(res["rgb"], res["ids"], res["conf"], res["is_prob"])

# -------------------------------- video --------------------------------
with tab_vid:
    vid = st.file_uploader("Video del recorrido", type=["mp4", "avi", "mov", "mkv"], key="vid")
    c1, c2 = st.columns(2)
    step = c1.number_input("Analizar 1 de cada N fotogramas", 1, 300, 30)
    max_frames = c2.slider("Máximo de fotogramas a analizar", 5, 60, 20)
    if vid is not None and st.button("Analizar video"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vid.name)[1])
        tmp.write(vid.read())
        tmp.close()
        try:
            frames, fps, _ = sample_video_frames(tmp.name, int(step), max_frames, max_w)
        finally:
            os.unlink(tmp.name)
        results, bar = [], st.progress(0.0)
        for i, f in enumerate(frames):
            ids, conf, is_prob = analyze(f)
            results.append(dict(rgb=f, ids=ids, conf=conf))
            bar.progress((i + 1) / len(frames))
        st.session_state["vid_res"] = dict(results=results, is_prob=is_prob, fps=fps, step=int(step))
    vres = st.session_state.get("vid_res")
    if vres and vres["results"]:
        n = len(vres["results"])
        idx = st.slider("Fotograma analizado", 0, n - 1, 0) if n > 1 else 0
        r = vres["results"][idx]
        show_results(r["rgb"], r["ids"], r["conf"], vres["is_prob"])
        per_frame = [summarize(x["ids"], x["conf"])[1] for x in vres["results"]]
        chart = pd.DataFrame(per_frame, index=[round(i * vres["step"] / vres["fps"], 1) for i in range(n)])
        chart.index.name = "segundo del video"
        st.subheader("Transitabilidad a lo largo del recorrido (% de píxeles)")
        st.line_chart(chart)
        st.write("Promedio del recorrido: " + " · ".join(f"{k} {v:.1f} %" for k, v in chart.mean().items()))
