"""
=========================================================
NAZALYST 

Dataset Management Pages

=========================================================
"""

import streamlit as st
import pandas as pd

from components.config import *
from components.cards import *
from components.sidebar import sidebar
from components.helpers import *

# ==========================================================
# PAGE CONFIG
# ==========================================================

setup_page()

load_css()

sidebar()

# ==========================================================
# LOAD SERVICES
# ==========================================================

updater = get_dataset_updater()

trainer = get_model_trainer()

metadata_service = get_metadata_service()

predictor = get_sentiment_predictor()

# ==========================================================
# LOAD INFORMATION
# ==========================================================

metadata = load_metadata()

training = load_training_info()

health = system_health()

status = get_service_status()

# ==========================================================
# SESSION STATE
# ==========================================================

if "update_result" not in st.session_state:

    st.session_state.update_result = None

if "training_result" not in st.session_state:

    st.session_state.training_result = None

# ==========================================================
# HEADER
# ==========================================================

section_title("📂 Dataset Management")

st.markdown(
"""
Halaman ini digunakan untuk mengelola dataset,
memperbarui data ulasan Google Play Store,
melakukan pelatihan ulang model serta
memantau status sistem NAZALYST.
"""
)

st.divider()

# ==========================================================
# DATASET INFORMATION
# ==========================================================

section_title("📊 Dataset Information")

c1, c2, c3, c4 = st.columns(4)

with c1:

    metric_card(

        "Total Review",

        format_number(

            metadata.get(

                "total_reviews",

                0

            )

        ),

        "📄"

    )

with c2:

    metric_card(

        "Average Rating",

        metadata.get(

            "average_rating",

            0

        ),

        "⭐",

        "#F59E0B"

    )

with c3:

    metric_card(

        "Last Update",

        str(

            metadata.get(

                "last_updated",

                "-"

            )

        )[:10],

        "📅",

        "#0066B3"

    )

with c4:

    model_status = (

        "READY"

        if status["model_exists"]

        else "NOT READY"

    )

    metric_card(

        "Model",

        model_status,

        "🤖",

        "#16A34A"

        if status["model_exists"]

        else "#DC2626"

    )

st.divider()

section_title("📋 Dataset Summary")

c1, c2, c3 = st.columns(3)

with c1:

    info_card(

        "Dataset",

        f"""

Total Review

{metadata.get("total_reviews",0):,}

Google Play Store

"""

    )

with c2:

    info_card(

        "Last Review",

        f"""

{metadata.get("latest_review","-")}

"""

    )

with c3:

    info_card(

        "Last Training",

        f"""

{training.get("last_training","Belum Pernah")}

"""

    )

st.divider()

# ==========================================================
# UPDATE DATASET
# ==========================================================

section_title("🔄 Update Dataset")

st.markdown("""
Perbarui dataset secara otomatis dari Google Play Store.
Sistem hanya akan menambahkan review baru yang belum
tersimpan pada dataset.
""")

progress_bar = st.progress(0)

status_placeholder = st.empty()

result_placeholder = st.empty()

update_btn = st.button(

    "🚀 Update Dataset",

    use_container_width=True,

    type="primary"

)

def update_progress(info):

    stage = info.get(

        "stage",

        ""

    )

    fetched = info.get(

        "fetched",

        0

    )

    new_reviews = info.get(

        "new_reviews",

        0

    )

    duplicate = info.get(

        "duplicate_reviews",

        0

    )

    page = info.get(

        "page",

        0

    )

    progress = min(

        page * 5,

        95

    )

    progress_bar.progress(progress)

    status_placeholder.info(

        f"""
**Stage**

{stage}

Page : {page}

Fetched : {fetched}

New Review : {new_reviews}

Duplicate : {duplicate}
"""

    )

if update_btn:

    with st.spinner(

        "Mengambil review terbaru..."

    ):

        try:

            result = updater.update_dataset(

                progress_callback=update_progress

            )

            st.session_state.update_result = result

            refresh_cache()

            progress_bar.progress(100)

            status_placeholder.success(

                "Update dataset selesai."

            )

        except Exception as e:

            progress_bar.empty()

            status_placeholder.error(

                str(e)

            )

if st.session_state.update_result is not None:

    result = st.session_state.update_result

    st.divider()

    section_title("📋 Update Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(

            "Review Baru",

            result.new_reviews,

            "🆕",

            "#16A34A"

        )

    with c2:

        metric_card(

            "Duplicate",

            result.duplicate_reviews,

            "♻️",

            "#F59E0B"

        )

    with c3:

        metric_card(

            "Dataset Lama",

            result.previous_total,

            "📄",

            "#0066B3"

        )

    with c4:

        metric_card(

            "Dataset Baru",

            result.current_total,

            "📈",

            "#16A34A"

        )

if (

    st.session_state.update_result is not None

    and

    not st.session_state.update_result.dataset.empty

):

    st.divider()

    section_title("👀 Preview Dataset Terbaru")

    st.dataframe(

        st.session_state.update_result.dataset.head(20),

        use_container_width=True,

        hide_index=True,

    )
    # ==========================================================
# RETRAIN MODEL
# ==========================================================

section_title("🧠 Retrain Model")

st.markdown("""
Setelah dataset berhasil diperbarui, model dapat
dilatih ulang agar menggunakan data terbaru.
""")

train_progress = st.progress(0)

train_status = st.empty()

train_btn = st.button(

    "🚀 Retrain Model",

    use_container_width=True,

    type="primary"

)

def train_progress_callback(info):

    stage = info.get(

        "stage",

        ""

    )

    progress = info.get(

        "progress",

        0

    )

    train_progress.progress(progress)

    train_status.info(

        f"""

### {stage}

Progress : **{progress}%**

"""

    )

if train_btn:

    with st.spinner(

        "Training model..."

    ):

        try:

            result = trainer.train(

                progress_callback=train_progress_callback

            )

            st.session_state.training_result = result

            refresh_cache()

            train_progress.progress(100)

            train_status.success(

                "Training berhasil."

            )

        except Exception as e:

            train_progress.empty()

            train_status.error(

                str(e)

            )

if st.session_state.training_result is not None:

    training = st.session_state.training_result

    st.divider()

    section_title("📊 Training Summary")

    c1, c2, c3 = st.columns(3)

    with c1:

        metric_card(

            "Status",

            training.status.upper(),

            "✅",

            "#16A34A"

        )

    with c2:

        metric_card(

            "Duration",

            f"{training.duration:.2f} sec",

            "⏱",

            "#0066B3"

        )

    with c3:

        metric_card(

            "Finished",

            training.finished_at.strftime(

                "%d-%m-%Y %H:%M"

            ),

            "📅",

            "#F59E0B"

        )
    st.divider()

section_title("🤖 Model Information")

model_info = predictor.summary()

c1, c2 = st.columns(2)

with c1:

    info_card(

        "Model",

        f"""

Support Vector Machine

TF-IDF

Classes :

{model_info['total_classes']}

"""

    )

with c2:

    info_card(

        "Prediction",

        f"""

Model :

{model_info['model_exists']}

Vectorizer :

{model_info['vectorizer_exists']}

Encoder :

{model_info['label_encoder_exists']}

"""

    )

# ==========================================================
# SYSTEM HEALTH
# ==========================================================

st.divider()

section_title("🟢 System Health")

health = system_health()

items = health["items"]

c1, c2 = st.columns(2)

with c1:

    status_card = []

    status_card.append(
        f"Dataset : {'🟢 Ready' if items['dataset_exists'] else '🔴 Missing'}"
    )

    status_card.append(
        f"Prediction Dataset : {'🟢 Ready' if items['prediction_dataset_exists'] else '🔴 Missing'}"
    )

    status_card.append(
        f"Metadata : {'🟢 Ready' if items['metadata_exists'] else '🔴 Missing'}"
    )

    info_card(

        "Dataset Status",

        "\n\n".join(status_card)

    )

with c2:

    model_card = []

    model_card.append(
        f"Model : {'🟢 Ready' if items['model_exists'] else '🔴 Missing'}"
    )

    model_card.append(
        f"Vectorizer : {'🟢 Ready' if items['vectorizer_exists'] else '🔴 Missing'}"
    )

    model_card.append(
        f"Label Encoder : {'🟢 Ready' if items['label_encoder_exists'] else '🔴 Missing'}"
    )

    info_card(

        "Model Status",

        "\n\n".join(model_card)

    )

# ==========================================================
# SYSTEM INFORMATION
# ==========================================================

st.divider()

section_title("⚙️ System Information")

c1, c2, c3 = st.columns(3)

with c1:

    info_card(

        "Application",

        f"""

Name

{APP_NAME}

Version

{APP_VERSION}

"""

    )

with c2:

    info_card(

        "Machine Learning",

        f"""

Model

{MODEL_NAME}

Feature

{FEATURE_EXTRACTION}

"""

    )

with c3:

    info_card(

        "Data Source",

        f"""

{DATA_SOURCE}

"""

    )

# ==========================================================
# INSIGHT
# ==========================================================

st.divider()

section_title("💡 Insight")

if health["healthy"]:

    success_card(

        "System Ready",

        """
Semua komponen utama sistem telah tersedia.

Dataset, metadata, model, vectorizer dan
label encoder siap digunakan.

NAZALYST siap melakukan analisis sentimen,
update dataset dan retraining model.
"""

    )

else:

    warning_card(

        "System Incomplete",

        """
Masih terdapat beberapa komponen sistem
yang belum tersedia.

Silakan lakukan update dataset atau
training model terlebih dahulu.
"""

    )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(

    f"""

NAZALYST © 2026 | Dashboard Analisis Sentimen MyPertamina

"""

)

