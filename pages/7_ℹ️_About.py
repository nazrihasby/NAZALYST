"""
=========================================================
NAZALYST

About Dashboard Pages

=========================================================
"""

import streamlit as st

from components.config import *
from components.cards import *
from components.sidebar import sidebar

# ==========================================================
# PAGE CONFIG
# ==========================================================

setup_page()
load_css()
sidebar()

# ==========================================================
# HEADER
# ==========================================================

section_title("ℹ️ About NAZALYST")

st.markdown("""
NAZALYST merupakan aplikasi berbasis **Streamlit**
yang dikembangkan untuk melakukan analisis sentimen
terhadap ulasan aplikasi **MyPertamina** menggunakan
algoritma **Support Vector Machine (SVM)**.

Dashboard ini dibuat sebagai implementasi penelitian
Tugas Akhir pada Program Studi Teknik Informatika.
""")

st.divider()

# ==========================================================
# PENELITIAN
# ==========================================================

section_title("📖 Informasi Penelitian")

about_card(

    "Judul Penelitian",

    """
Implementasi Analisis Sentimen pada Ulasan
Aplikasi MyPertamina Menggunakan
Algoritma Support Vector Machine
Berbasis Web Streamlit.
"""

)

st.divider()

section_title("⚙️ Metode")

c1,c2=st.columns(2)

with c1:

    info_card(

        "Machine Learning",

        """
Support Vector Machine

Linear Kernel

CalibratedClassifierCV
"""

    )

with c2:

    info_card(

        "Feature Extraction",

        """
TF-IDF Vectorizer

Unigram

Bigram

Trigram
"""

    )

    st.divider()

section_title("📊 Dataset")

info_card(

    "Dataset",

    """
Sumber :

Google Play Store

Kolom Dataset

• text

• label

• pred_label

• created_at

• score

• author

• id

• year
"""

)

st.divider()

section_title("🧰 Teknologi")

col1,col2,col3=st.columns(3)

with col1:

    info_card(

        "Backend",

        """
Python

Scikit-learn

Pandas
"""

    )

with col2:

    info_card(

        "Visualization",

        """
Plotly

Matplotlib

WordCloud
"""

    )

with col3:

    info_card(

        "Framework",

        """
Streamlit

Joblib

NumPy
"""

    )

st.divider()

section_title("👨‍💻 Developer")

success_card(

    "Developer",

    """
Naz Hasby

Program Studi Teknik Informatika

Fakultas Teknologi Industri

Universitas Trisakti
"""

)

st.divider()

section_title("🙏 Ucapan Terima Kasih")

st.markdown("""

Terima kasih kepada:

- Allah SWT

- Orang Tua

- Dosen Pembimbing

- Program Studi Teknik Informatika

- Universitas Trisakti

- Seluruh pihak yang telah membantu
penelitian ini.

""")

st.divider()

section_title("📌 Versi Aplikasi")

metric_card(

    title="Version",

    value="NAZALYST © 2026 | Dashboard Analisis Sentimen MyPertamina",

    icon="🚀",

    color="#0066B3"

)

