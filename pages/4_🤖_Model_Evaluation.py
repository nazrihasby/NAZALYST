"""
=========================================================
NAZALYST

Model Evaluation Dashboard Pages

=========================================================
"""

import streamlit as st

from components.config import *
from components.cards import *
from components.metrics import *
from components.charts import *
from components.sidebar import sidebar

# =====================================================
# CONFIG
# =====================================================

setup_page()
load_css()
sidebar()

# =====================================================
# LOAD DATA
# =====================================================

df = load_dataset()

result = evaluate_dataframe(df)

# =====================================================
# HEADER
# =====================================================

section_title("🤖 Model Evaluation")

st.markdown(
"""
Halaman ini menampilkan performa model
Support Vector Machine berdasarkan hasil
prediksi terhadap dataset.
"""
)

st.divider()

# =====================================================
# KPI
# =====================================================

c1,c2,c3,c4=st.columns(4)

with c1:

    metric_card(

        "Accuracy",

        f"{result['accuracy']*100:.2f}%",

        "🎯",

        "#0066B3"

    )

with c2:

    metric_card(

        "Precision",

        f"{result['precision']*100:.2f}%",

        "📈",

        "#16A34A"

    )

with c3:

    metric_card(

        "Recall",

        f"{result['recall']*100:.2f}%",

        "📊",

        "#F59E0B"

    )

with c4:

    metric_card(

        "F1 Score",

        f"{result['f1']*100:.2f}%",

        "🤖",

        "#D71920"

    )

st.divider()

# =====================================================
# ACCURACY
# =====================================================

section_title("🎯 Accuracy")

fig = accuracy_gauge(

    result["accuracy"]

)

st.plotly_chart(

    fig,

    use_container_width=True

)

st.divider()

# =====================================================
# CONFUSION MATRIX
# =====================================================

section_title("📊 Confusion Matrix")

fig = confusion_matrix_chart(

    result["confusion_matrix"],

    result["labels"]

)

st.plotly_chart(

    fig,

    use_container_width=True

)

st.divider()

# =====================================================
# REPORT
# =====================================================

section_title("📄 Classification Report")

fig = classification_table(

    result["classification_report"]

)

st.plotly_chart(

    fig,

    use_container_width=True

)

st.divider()

section_title("🧠 Insight")

acc = result["accuracy"]

if acc >= 0.90:

    success_card(

        "Model Sangat Baik",

        "Model memiliki performa yang sangat tinggi."

    )

elif acc >= 0.80:

    success_card(

        "Model Baik",

        "Model sudah layak digunakan."

    )

elif acc >= 0.70:

    warning_card(

        "Model Cukup",

        "Model masih dapat ditingkatkan."

    )

else:

    error_card(

        "Model Kurang",

        "Model perlu dilakukan pelatihan ulang."

    )

st.divider()

section_title("📋 Ringkasan")

st.markdown(f"""

- Accuracy : **{result['accuracy']*100:.2f}%**

- Precision : **{result['precision']*100:.2f}%**

- Recall : **{result['recall']*100:.2f}%**

- F1 Score : **{result['f1']*100:.2f}%**

""")