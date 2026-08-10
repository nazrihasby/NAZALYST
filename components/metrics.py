"""
metrics.py Nazalyst
"""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# EVALUATE DATAFRAME
# ==========================================================

def evaluate_dataframe(df):

    """
    Menghitung seluruh metrik evaluasi
    berdasarkan kolom:

    label
    pred_label
    """

    if "label" not in df.columns:
        raise ValueError("Kolom 'label' tidak ditemukan.")

    if "pred_label" not in df.columns:
        raise ValueError("Kolom 'pred_label' tidak ditemukan.")

    y_true = df["label"].astype(str)

    y_pred = df["pred_label"].astype(str)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    report = classification_report(

        y_true,

        y_pred,

        output_dict=True,

        zero_division=0

    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .round(4)
    )

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "confusion_matrix": cm,

        "classification_report": report_df,

        "labels": sorted(
            list(
                set(y_true) | set(y_pred)
            )
        )

    }