"""
=========================================================
NAZALYST 

Sentiment Analysis Pages

=========================================================
"""

import streamlit as st

from components.config import *
from components.cards import *
from components.charts import *
from components.helpers import *
from components.sidebar import sidebar
from components.insight import sentiment_insight

# ======================================================
# CONFIG
# ======================================================

setup_page()
load_css()
sidebar()

# ======================================================
# LOAD DATA
# ======================================================

df = load_dataset()

section_title("📈 Sentiment Analysis")

st.write(
    "Analisis distribusi sentimen berdasarkan hasil klasifikasi model."
)

st.divider()

# ======================================================
# FILTER
# ======================================================

st.subheader("🎛 Filter Data")

c1, c2 = st.columns(2)

with c1:

    sentiment = st.multiselect(
        "Sentimen",
        options=sorted(df["pred_label"].unique()),
        default=sorted(df["pred_label"].unique())
    )

with c2:

    keyword = st.text_input(
        "Keyword"
    )

filtered = df.copy()

filtered = filtered[
    filtered["pred_label"].isin(sentiment)
]

if keyword:

    filtered = filter_keyword(
        filtered,
        keyword
    )

summary = sentiment_distribution(filtered)

st.divider()

# ======================================================
# KPI
# ======================================================

c1,c2,c3,c4=st.columns(4)

with c1:

    metric_card(
        "Total",
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

# ======================================================
# BAR & DONUT
# ======================================================

left,right=st.columns(2)

with left:

    st.plotly_chart(

        sentiment_bar(filtered),

        use_container_width=True

    )

with right:

    st.plotly_chart(

        sentiment_donut(filtered),

        use_container_width=True

    )

st.divider()

# ======================================================
# RATING
# ======================================================

section_title("⭐ Rating Distribution")

fig = rating_distribution(filtered)

if fig:

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ======================================================
# STACKED
# ======================================================

section_title("📊 Distribusi Tahunan")

fig = stacked_sentiment(filtered)

if fig:

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ======================================================
# TREND
# ======================================================

section_title("📈 Trend Sentiment")

fig = sentiment_trend(filtered)

if fig:

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ======================================================
# HEATMAP
# ======================================================

section_title("🔥 Heatmap Rating vs Sentiment")

fig = sentiment_heatmap(filtered)

if fig:

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ======================================================
# INSIGHT
# ======================================================

section_title("🧠 Insight")

sentiment_insight(summary)

st.divider()

# ======================================================
# DATA
# ======================================================

section_title("📄 Preview Data")

st.dataframe(

    filtered.head(50),

    use_container_width=True

)