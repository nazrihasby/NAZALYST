"""
=========================================================
NAZALYST
Landing Page
=========================================================
"""

import streamlit as st
from components.cards import metric_card
from components.config import *
from components.sidebar import sidebar


# ==============================
# PAGE CONFIG
# ==============================
setup_page()
load_css()
sidebar()

# ==============================
# HERO
# ==============================
st.markdown(
"""
<div class="hero">

<h1>📊 NAZALYST</h1>

<h3>
Interactive Dashboard Analisis Sentimen
</h3>

<p>
Analisis Sentimen Ulasan Aplikasi MyPertamina
Menggunakan Support Vector Machine (SVM)
dan TF-IDF
</p>

</div>
""",
unsafe_allow_html=True
)

st.write("")

# ==============================
# DASHBOARD DESCRIPTION
# ==============================
st.markdown(
"""
Selamat datang di **NAZALYST Dashboard**.

Dashboard ini dikembangkan untuk membantu proses analisis
sentimen terhadap ulasan pengguna aplikasi **MyPertamina**
menggunakan algoritma **Support Vector Machine (SVM)**.

Melalui dashboard ini pengguna dapat:

- 📈 Melihat distribusi sentimen
- 📊 Melihat statistik dataset
- ☁️ Melakukan analisis teks
- 🤖 Melihat performa model
- 🔍 Melakukan prediksi sentimen secara langsung
- 📂 Memantau dataset terbaru dari google play store dan bisa me retrain dataset terbaru
- ℹ️ Mengetahui informasi pembuat aplikasi, tujuan, dan teknologi yang digunakan

Silakan gunakan menu di sebelah kiri untuk mulai
menjelajahi dashboard.
"""
)

st.divider()

# ==============================
# QUICK INFO
# ==============================
col1,col2,col3,col4=st.columns(4)

with col1:

    metric_card(

        title="Total Review",

        value="198.514",

        icon="📄"

    )

with col2:

    metric_card(

        title="Positif",

        value="18.71%",

        icon="😊",

        color="#16A34A"

    )

with col3:

    metric_card(

        title="Negatif",

        value="81.22%",

        icon="😡",

        color="#DC2626"

    )

with col4:

    metric_card(

        title="Model",

        value="SVM",

        icon="🤖",

        color="#0066B3"

    )

st.divider()

# ==============================
# FEATURES
# ==============================
st.subheader("🚀 Fitur Dashboard")

feature1, feature2, feature3 = st.columns(3)

with feature1:

    st.markdown(
        """
### 📈 Sentiment Analysis

Menampilkan distribusi sentimen
positif, negatif, dan netral
secara interaktif.
"""
    )

with feature2:

    st.markdown(
        """
### ☁️ Text Analytics

WordCloud

Top Words

Bigram

Trigram
"""
    )

with feature3:

    st.markdown(
        """
### 🤖 Machine Learning

Evaluasi Model

Confusion Matrix

Live Prediction
"""
    )

st.divider()

# ==============================
# FOOTER
# ==============================
st.caption(
    "NAZALYST © 2026 | Dashboard Analisis Sentimen MyPertamina"
)