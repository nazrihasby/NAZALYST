# 📊 NAZALYST — Sentiment Analysis App
# 1. Overview
NAZALYST adalah aplikasi berbasis Streamlit untuk melakukan analisis sentimen ulasan aplikasi MyPertamina menggunakan algoritma Support Vector Machine (SVM) dengan fitur pembaruan dataset otomatis, pelatihan ulang model, prediksi sentimen, dan visualisasi interaktif.

# 2. Features
✔ Google Play Review Scraper
✔ Dataset Management
✔ Duplicate Detection
✔ Automatic Retraining
✔ TF-IDF Feature Extraction
✔ Support Vector Machine
✔ Sentiment Prediction
✔ Interactive Dashboard

# 3. Project Structure
NAZALYST/
├── app.py

├── generate_predictions.py

├── preprocess.py

├── README.md

├── requirements.txt

├── train.py

├──.streamlit/
        └── config.toml

├── components/
        ├── cards.py
        ├── charts.py
        ├── config.py
        ├── helpers.py
        ├── insight.py
        ├── metrics.py
        └── sidebar.py

├── data/
        ├── dataset.csv
        └── hasil_prediksi.csv

├── Envdal/
        ├── etc
        ├── Include
        ├── Lib
        ├── Scripts
        ├── share
        └── pyenv.cfg

├── models/
        ├── label_encoder.pkl
        ├── model.pkl
        └── vectorizer.pkl

├── pages/
        ├── 1_Overview.py
        ├── 2_Sentiment_Analysis.py
        ├── 3_Text_Analytics.py
        ├── 4_Model_Evaluation.py
        ├── 5_Prediction.py
        ├── 6_Dataset_Management.py
        └── 7_About.py

├── services/
        ├── metada.py
        ├── predictor.py
        ├── scraper.py
        ├── trainer.py
        └── updater.py

└── styles/
        └── style.css

# 4. Instalastion
git clone ...

cd NAZALYST

python -m venv .venv

pip install -r requirements.txt


# 5. How To Run

- Access Environment
Envdal\scripts\activate.bat

- Running the streamlit
streamlit run app.py

- Training Model 
python train.py --csv data/dataset.csv

- Generate Prediction Terbaru
python generate_predictions.py



# 6. Requirements System inside the requirements.txt
Python
Streamlit
Scikit-learn
Plotly
NLTK
Sastrawi
Google Play Scraper
Pandas
NumPy
WordCloud


# 7. Lisensi
Proyek ini dibuat untuk tujuan akademik & penelitian. Gunakan dengan memperhatikan etika data & privasi.

Nazri Hasby - Sistem Informasi

Universitas Trisakti








4. Installation

git clone ...

cd NAZALYST

python -m venv .venv

pip install -r requirements.txt

5. How to Run

Envdal\scripts\activate.bat

streamlit run app.py

Melatih Model
python train.py --csv data/dataset.csv

Generate Prediction Terbaru
python generate_predictions.py

6. Dashboard Workflow

Google Play Reviews
        │
        ▼
Dataset Update
        │
        ▼
dataset.csv
        │
        ▼
Retrain Model
        │
        ▼
Prediction
        │
        ▼
Dashboard Visualization

7. Requirement

Python

Streamlit

Scikit-learn

Plotly

NLTK

Sastrawi

Google Play Scraper

Pandas

NumPy

WordCloud

8. Troubleshooting
ModuleNotFoundError

↓

pip install -r requirements.txt

----------------------

Model not found

↓

Run train.py

----------------------

Dataset empty

↓

Run Update Dataset



















✔ Trend Analysis

✔ Model Evaluation

✔ Metadata Management