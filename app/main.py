"""
main.py - App Streamlit para detección de mensajes de odio en comentarios de YouTube.
Proyecto: NLP - Análisis de Sentimientos / Hate Speech Detection
"""

import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import clean_text

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Rutas a los artefactos del modelo ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


# ── Carga del modelo (cacheado para eficiencia) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Carga el modelo y el vectorizador TF-IDF desde disco."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str, model, vectorizer):
    """Devuelve etiqueta y probabilidad para un texto dado."""
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    label = model.predict(X)[0]
    # Probabilidades disponibles solo para modelos que las soporten
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = proba[1] if label == 1 else proba[0]
    else:
        # LinearSVC → usar decision_function como proxy
        score = model.decision_function(X)[0]
        # Normalizar a [0, 1] con sigmoide simple
        import math
        confidence = 1 / (1 + math.exp(-score)) if label == 1 else 1 / (1 + math.exp(score))
    return int(label), float(confidence)


# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .toxic-box {
            background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
            border-left: 5px solid #ff4b4b;
            border-radius: 8px;
            padding: 18px 22px;
            margin: 10px 0;
        }
        .safe-box {
            background: linear-gradient(135deg, #00c85322, #00c85344);
            border-left: 5px solid #00c853;
            border-radius: 8px;
            padding: 18px 22px;
            margin: 10px 0;
        }
        .metric-card {
            background: #1e1e2e;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }
        .stTextArea textarea { font-size: 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/youtube-play.png", width=70)
    st.title("🛡️ Hate Speech\nDetector")
    st.markdown("---")
    st.markdown(
        """
        ### ¿Cómo funciona?
        1. Escribe o pega un comentario de YouTube.
        2. El modelo analiza el texto automáticamente.
        3. Recibe la clasificación y nivel de confianza.

        ---
        ### Modelo
        - **Vectorización:** TF-IDF
        - **Clasificador:** Logistic Regression / LinearSVC
        - **Dataset:** YouToxic English (1 000 muestras)

        ---
        ### Etiquetas
        | Etiqueta | Significado |
        |---|---|
        | ✅ No tóxico | Comentario seguro |
        | 🚨 Tóxico | Contiene odio/abuso |
        """
    )
    st.markdown("---")
    st.caption("Proyecto académico · NLP · 2025")

# ── Título principal ──────────────────────────────────────────────────────────
st.title("🛡️ Detector de Mensajes de Odio en YouTube")
st.markdown(
    "Analiza comentarios de YouTube y detecta si contienen lenguaje tóxico, "
    "odio o acoso utilizando un modelo de Machine Learning entrenado con datos reales."
)
st.markdown("---")

# ── Carga de artefactos ───────────────────────────────────────────────────────
with st.spinner("Cargando modelo..."):
    model, vectorizer = load_artifacts()

if model is None:
    st.error(
        "⚠️ No se encontraron los artefactos del modelo en `models/`. "
        "Por favor, ejecuta primero el notebook `02_Modelado.ipynb` para entrenar y guardar el modelo."
    )
    st.stop()

# ── Tabs principales ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Análisis Individual", "📋 Análisis por Lotes"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Análisis individual
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1.2, 1], gap="large")

    with col_input:
        st.subheader("✍️ Introduce un comentario")
        user_text = st.text_area(
            label="Comentario",
            placeholder="Escribe aquí el comentario de YouTube que quieres analizar...",
            height=180,
            label_visibility="collapsed",
        )

        example_col1, example_col2 = st.columns(2)
        with example_col1:
            if st.button("💬 Ejemplo seguro", use_container_width=True):
                user_text = "This video is amazing! Really well explained, thank you so much."
                st.session_state["example_text"] = user_text
        with example_col2:
            if st.button("⚠️ Ejemplo tóxico", use_container_width=True):
                user_text = "You are so stupid and ugly, nobody wants to see your disgusting face."
                st.session_state["example_text"] = user_text

        # Si se cargó un ejemplo, mostrarlo en el text_area
        if "example_text" in st.session_state and not user_text:
            user_text = st.session_state["example_text"]

        analyze_btn = st.button("🔍 Analizar comentario", type="primary", use_container_width=True)

    with col_result:
        st.subheader("📊 Resultado")

        if analyze_btn or ("example_text" in st.session_state and user_text):
            if not user_text.strip():
                st.warning("Por favor, introduce un comentario antes de analizar.")
            else:
                with st.spinner("Analizando..."):
                    label, confidence = predict(user_text, model, vectorizer)

                if label == 1:
                    st.markdown(
                        f"""
                        <div class="toxic-box">
                            <h2 style="color:#ff4b4b; margin:0;">🚨 TÓXICO</h2>
                            <p style="font-size:16px; margin:8px 0 0 0;">
                                Este comentario ha sido clasificado como <b>tóxico o con lenguaje de odio</b>.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="safe-box">
                            <h2 style="color:#00c853; margin:0;">✅ NO TÓXICO</h2>
                            <p style="font-size:16px; margin:8px 0 0 0;">
                                Este comentario parece <b>seguro y sin lenguaje ofensivo</b>.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.metric(
                    label="Confianza del modelo",
                    value=f"{confidence * 100:.1f}%",
                    help="Probabilidad estimada de que la clasificación sea correcta.",
                )

                # Gráfico de gauge simple con matplotlib
                fig, ax = plt.subplots(figsize=(4, 0.5))
                color = "#ff4b4b" if label == 1 else "#00c853"
                ax.barh(0, confidence, color=color, height=0.4)
                ax.barh(0, 1 - confidence, left=confidence, color="#e0e0e033", height=0.4)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.axis("off")
                fig.patch.set_alpha(0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # Texto preprocesado
                with st.expander("🔬 Ver texto preprocesado"):
                    cleaned = clean_text(user_text)
                    st.code(cleaned if cleaned else "(texto vacío tras preprocesamiento)", language=None)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Análisis por lotes
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📋 Analiza múltiples comentarios a la vez")
    st.markdown(
        "Sube un archivo CSV con una columna llamada **`Text`** que contenga los comentarios a analizar."
    )

    uploaded_file = st.file_uploader(
        "Sube tu CSV", type=["csv"], help="El archivo debe tener al menos una columna 'Text'."
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            st.stop()

        if "Text" not in df_upload.columns:
            st.error("⚠️ El CSV debe contener una columna llamada **`Text`**.")
        else:
            st.success(f"✅ Archivo cargado: **{len(df_upload)} comentarios**.")

            with st.spinner("Analizando todos los comentarios..."):
                df_upload["Predicción"] = df_upload["Text"].apply(
                    lambda t: "🚨 Tóxico" if predict(str(t), model, vectorizer)[0] == 1 else "✅ No tóxico"
                )
                df_upload["Confianza (%)"] = df_upload["Text"].apply(
                    lambda t: round(predict(str(t), model, vectorizer)[1] * 100, 1)
                )

            # Estadísticas rápidas
            n_toxic = (df_upload["Predicción"] == "🚨 Tóxico").sum()
            n_safe = len(df_upload) - n_toxic

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total comentarios", len(df_upload))
            col_b.metric("🚨 Tóxicos", n_toxic)
            col_c.metric("✅ No tóxicos", n_safe)

            # Gráfico de distribución
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.pie(
                [n_safe, n_toxic],
                labels=["No tóxico", "Tóxico"],
                colors=["#00c853", "#ff4b4b"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops=dict(width=0.6),
            )
            ax2.set_title("Distribución de predicciones", fontsize=12)
            fig2.patch.set_alpha(0)
            st.pyplot(fig2, use_container_width=False)
            plt.close(fig2)

            # Tabla de resultados
            st.subheader("Resultados detallados")
            st.dataframe(
                df_upload[["Text", "Predicción", "Confianza (%)"]],
                use_container_width=True,
                hide_index=True,
            )

            # Descarga de resultados
            csv_result = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar resultados CSV",
                data=csv_result,
                file_name="resultados_predicciones.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        # Instrucciones y ejemplo de descarga
        st.info(
            "💡 **Formato esperado del CSV:**\n\n"
            "```\nText\n"
            "This video is great!\n"
            "You are a terrible person.\n"
            "I love this channel.\n```"
        )
