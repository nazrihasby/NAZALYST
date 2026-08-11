"""
=========================================================
NAZALYST

Text Analytics Dashboard Pages

=========================================================
"""

import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from components.config import *
from components.cards import *
from components.charts import *
from components.helpers import *
from components.sidebar import sidebar

# =====================================================
# PAGE
# =====================================================

setup_page()
load_css()
sidebar()

# =====================================================
# LOAD DATA
# =====================================================

df = load_dataset()

# =====================================================
# HEADER
# =====================================================

section_title("☁️ Text Analytics")

st.markdown(
"""
Halaman ini digunakan untuk melakukan analisis teks
berdasarkan hasil klasifikasi sentimen.
"""
)

st.divider()

# =====================================================
# FILTER
# =====================================================

sentiment = st.selectbox(

    "Pilih Sentimen",

    [

        "Semua",

        "positif",

        "negatif",

        "netral"

    ]

)

if sentiment != "Semua":

    df = filter_sentiment(

        df,

        [sentiment]

    )

st.divider()

# =====================================================
# WORD CLOUD
# =====================================================

section_title("☁️ Word Cloud")

# Pastikan kolom text tersedia
if "text" not in df.columns:
    st.error("Kolom 'text' tidak ditemukan pada dataset.")
    st.stop()

# Bersihkan nilai kosong dan pastikan seluruh data berupa string
text_series = (
    df["text"]
    .fillna("")
    .astype(str)
)

# Batasi jumlah review untuk menjaga penggunaan memory
# pada Streamlit Cloud
max_wordcloud_rows = 20000

if len(text_series) > max_wordcloud_rows:
    text_series = text_series.sample(
        n=max_wordcloud_rows,
        random_state=42
    )

text = " ".join(text_series.tolist())

if text.strip() == "":
    st.warning("Tidak ada data.")
else:
    wc = WordCloud(
        width=1200,
        height=500,
        background_color="white",
        colormap="Reds"
    ).generate(text)

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    ax.imshow(
        wc,
        interpolation="bilinear"
    )

    ax.axis("off")

    st.pyplot(fig)

    plt.close(fig)

    # =====================================================
# TOP WORD
# =====================================================

section_title("📚 Top 20 Kata")

top = top_words(

    df,

    20

)

fig = word_frequency(

    top.set_index(

        "Kata"

    )["Frekuensi"],

    "Top 20 Kata"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# =====================================================
# BIGRAM
# =====================================================

section_title("📈 Bigram")

fig = bigram_chart(df)

st.plotly_chart(

    fig,

    use_container_width=True

)

# =====================================================
# TRIGRAM
# =====================================================

section_title("📊 Trigram")

fig = trigram_chart(df)

st.plotly_chart(

    fig,

    use_container_width=True

)

# =====================================================
# PREVIEW
# =====================================================

section_title("📄 Dataset")

st.dataframe(

    df.head(

        30

    ),

    use_container_width=True

)