"""
=========================================================
NAZALYST 

Overview Dashboard Pages

=========================================================
"""

import streamlit as st

from components.config import *
from components.cards import *
from components.charts import *
from components.helpers import *
from components.insight import *
from components.sidebar import sidebar

# ==========================================================
# PAGE CONFIG
# ==========================================================

setup_page()

load_css()

sidebar()

# ==========================================================
# LOAD DATA
# ==========================================================

df = load_dataset()

summary = sentiment_distribution(df)

# ==========================================================
# HEADER
# ==========================================================

section_title("📊 Dashboard Overview")

st.markdown(
"""
Selamat datang pada Dashboard Analisis Sentimen
Aplikasi **MyPertamina**.

Halaman ini menampilkan ringkasan dataset,
distribusi sentimen serta insight utama hasil
analisis.
"""
)

st.divider()

# ==========================================================
# KPI
# ==========================================================

c1,c2,c3,c4=st.columns(4)

with c1:

    metric_card(

        "Total Review",

        format_number(summary["total"]),

        "📄"

    )

with c2:

    metric_card(

        "Positif",

        format_percent(summary["positif_percent"]),

        "😊",

        "#16A34A"

    )

with c3:

    metric_card(

        "Negatif",

        format_percent(summary["negatif_percent"]),

        "😡",

        "#DC2626"

    )

with c4:

    metric_card(

        "Netral",

        format_percent(summary["netral_percent"]),

        "😐",

        "#0EA5E9"

    )

st.divider()

# ==========================================================
# DISTRIBUSI SENTIMEN
# ==========================================================

section_title("📈 Distribusi Sentimen")

left,right=st.columns(2)

with left:

    fig=sentiment_bar(df)

    st.plotly_chart(

        fig,

        use_container_width=True

    )

with right:

    fig=sentiment_donut(df)

    st.plotly_chart(

        fig,

        use_container_width=True

    )

st.divider()

# ==========================================================
# DISTRIBUSI RATING
# ==========================================================

section_title("⭐ Distribusi Rating")

fig=rating_distribution(df)

if fig:

    st.plotly_chart(

        fig,

        use_container_width=True

    )

else:

    st.info(

        "Kolom score tidak ditemukan."

    )

st.divider()

# ==========================================================
# TREND DATA
# ==========================================================

section_title("📅 Trend Data")

fig=yearly_sentiment(df)

if fig:

    st.plotly_chart(

        fig,

        use_container_width=True

    )

else:

    st.info(

        "Kolom year belum tersedia."

    )

st.divider()

# ==========================================================
# INSIGHT
# ==========================================================

section_title("🧠 Insight")

sentiment_insight(summary)

st.divider()

section_title("📋 Ringkasan Dataset")

c1,c2,c3=st.columns(3)

with c1:

    info_card(

        "Dataset",

        f"""
Jumlah Review

{summary['total']:,}

Google Play Store
"""

    )

with c2:

    info_card(

        "Model",

        """
Support Vector Machine

TF-IDF

Linear Kernel
"""

    )

with c3:

    info_card(

        "Output",

        """
Positif

Negatif

Netral
"""

    )

st.divider()

section_title("👀 Preview Dataset")

st.dataframe(

    df.head(20),

    use_container_width=True

)