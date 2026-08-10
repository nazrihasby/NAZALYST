"""
Sidebar.py Nazalyst
"""
import streamlit as st

def sidebar():

    with st.sidebar:

        st.markdown("# 📊 NAZALYST")

        st.caption(
            "Interactive Sentiment Dashboard"
        )

        st.divider()

        st.markdown(
            """
### 📁 Dataset

Google Play Store

MyPertamina
"""
        )

        st.divider()

        st.markdown(
            """
### 🤖 Model

Support Vector Machine

TF-IDF
"""
        )

        st.divider()

        st.success(
            "Gunakan menu Pages di sebelah kiri untuk berpindah halaman."
        )

        st.divider()

        st.caption(
            "NAZALYST © 2026 | Dashboard Analisis Sentimen MyPertamina"
        )