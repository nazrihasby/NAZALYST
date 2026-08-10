"""
=========================================================
NAZALYST

Prediction Dashboard Pages

=========================================================
"""

import numpy as np
import pandas as pd
import streamlit as st

from components.config import setup_page, load_css, load_model
from components.sidebar import sidebar
from components.cards import (
    section_title,
    metric_card,
    success_card,
    warning_card,
    error_card
)
from components.charts import (
    probability_chart,
    accuracy_gauge
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

setup_page()
load_css()
sidebar()

# ==========================================================
# LOAD MODEL
# ==========================================================

model, vectorizer, label_encoder = load_model()

# ==========================================================
# SESSION STATE
# ==========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ==========================================================
# HEADER
# ==========================================================

section_title("🔍 Prediksi Sentimen")

st.write(
    """
Masukkan ulasan pengguna kemudian sistem akan
memprediksi sentimen menggunakan model
Support Vector Machine (LinearSVC + Calibration).
"""
)

st.divider()

# ==========================================================
# INPUT
# ==========================================================

review = st.text_area(
    "Masukkan Review",
    height=180,
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan..."
)

predict_btn = st.button(
    "🚀 Prediksi",
    use_container_width=True
)

# ==========================================================
# PROCESS
# ==========================================================

if predict_btn:

    if review.strip() == "":

        st.warning(
            "Silakan masukkan review terlebih dahulu."
        )
        st.stop()

    # TF-IDF
    X = vectorizer.transform([review])

    # Prediksi
    pred = model.predict(X)[0]

    # Decode label jika menggunakan LabelEncoder
    try:
        label = label_encoder.inverse_transform([pred])[0]
    except Exception:
        label = str(pred)

    # Probabilitas
    proba = model.predict_proba(X)[0]

    # Confidence
    confidence = float(np.max(proba))

    # Warna
    if label.lower() == "positif":

        color = "#16A34A"
        emoji = "😊"

    elif label.lower() == "negatif":

        color = "#DC2626"
        emoji = "😡"

    else:

        color = "#0EA5E9"
        emoji = "😐"

    # ==========================================================
    # HASIL PREDIKSI
    # ==========================================================

    st.divider()

    section_title("🎯 Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:

        metric_card(
            title="Prediction",
            value=label.upper(),
            icon=emoji,
            color=color
        )

    with col2:

        metric_card(
            title="Confidence",
            value=f"{confidence * 100:.2f}%",
            icon="🎯",
            color=color
        )

    # ==========================================================
    # PROBABILITY
    # ==========================================================

    st.divider()

    section_title("📊 Probabilitas Setiap Kelas")

    try:

        labels = list(label_encoder.classes_)

    except Exception:

        labels = ["negatif", "netral", "positif"]

    fig = probability_chart(
        labels,
        proba
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================================================
    # GAUGE
    # ==========================================================

    st.divider()

    section_title("📈 Confidence Gauge")

    fig = accuracy_gauge(confidence)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================================================
    # INTERPRETASI
    # ==========================================================

    st.divider()

    section_title("💡 Interpretasi")

    if label.lower() == "positif":

        success_card(
            "Sentimen Positif",
            """
Model memprediksi bahwa ulasan memiliki
kecenderungan sentimen positif.

Hal ini menunjukkan pengguna memberikan
respon yang baik terhadap aplikasi.
"""
        )

    elif label.lower() == "negatif":

        error_card(
            "Sentimen Negatif",
            """
Model memprediksi bahwa ulasan memiliki
kecenderungan sentimen negatif.

Kemungkinan besar ulasan berisi keluhan,
kritik atau pengalaman yang kurang baik.
"""
        )

    else:

        warning_card(
            "Sentimen Netral",
            """
Model memprediksi bahwa ulasan bersifat
netral dan tidak memiliki kecenderungan
positif maupun negatif yang dominan.
"""
        )

    # ==========================================================
    # HISTORY
    # ==========================================================

    st.session_state.prediction_history.append(

        {

            "Review": review,

            "Prediction": label,

            "Confidence (%)": round(confidence * 100, 2)

        }

    )

    st.divider()

    section_title("🕒 Riwayat Prediksi")

    history = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

    # ==========================================================
    # DOWNLOAD
    # ==========================================================

    csv = history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download History CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

else:

    st.info(
        """
Masukkan review pada kotak teks di atas,
kemudian klik tombol **Prediksi**
untuk melihat hasil klasifikasi sentimen.
"""
    )