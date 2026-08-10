"""
Charts.py Nazalyst
"""
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from components.config import COLOR_MAP


# =========================================================
# GLOBAL TEMPLATE
# =========================================================

def apply_layout(fig, title=""):

    fig.update_layout(

        title=dict(
            text=title,
            x=0.5,
            font=dict(
                size=20
            )
        ),

        template="plotly_white",

        paper_bgcolor="white",

        plot_bgcolor="white",

        height=450,

        legend_title=None,

        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20
        ),

        font=dict(
            family="Inter",
            size=14
        )
    )

    return fig


# =========================================================
# SENTIMENT BAR CHART
# =========================================================

def sentiment_bar(df):

    counts = (
        df["pred_label"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [

        "Sentimen",

        "Jumlah"

    ]

    fig = px.bar(

        counts,

        x="Sentimen",

        y="Jumlah",

        color="Sentimen",

        text="Jumlah",

        color_discrete_map=COLOR_MAP

    )

    fig.update_traces(

        textposition="outside",

        marker_line_width=0

    )

    fig.update_xaxes(

        title=""

    )

    fig.update_yaxes(

        title="Jumlah Review"

    )

    return apply_layout(

        fig,

        "Distribusi Sentimen"

    )


# =========================================================
# DONUT CHART
# =========================================================

def sentiment_donut(df):

    counts = (
        df["pred_label"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [

        "Sentimen",

        "Jumlah"

    ]

    fig = px.pie(

        counts,

        names="Sentimen",

        values="Jumlah",

        hole=.60,

        color="Sentimen",

        color_discrete_map=COLOR_MAP

    )

    fig.update_traces(

        textinfo="percent+label"

    )

    return apply_layout(

        fig,

        "Proporsi Sentimen"

    )


# =========================================================
# RATING DISTRIBUTION
# =========================================================

def rating_distribution(df):

    if "score" not in df.columns:

        return None

    rating = (

        df["score"]

        .value_counts()

        .sort_index()

        .reset_index()

    )

    rating.columns = [

        "Rating",

        "Jumlah"

    ]

    fig = px.bar(

        rating,

        x="Rating",

        y="Jumlah",

        text="Jumlah",

        color="Rating",

        color_continuous_scale="Reds"

    )

    fig.update_traces(

        textposition="outside"

    )

    fig.update_xaxes(

        dtick=1

    )

    return apply_layout(

        fig,

        "Distribusi Rating"

    )


# =========================================================
# YEARLY SENTIMENT
# =========================================================

def yearly_sentiment(df):

    if "year" not in df.columns:

        return None

    yearly = (

        df.groupby(

            [

                "year",

                "pred_label"

            ]

        )

        .size()

        .reset_index(

            name="Jumlah"

        )

    )

    fig = px.bar(

        yearly,

        x="year",

        y="Jumlah",

        color="pred_label",

        barmode="stack",

        color_discrete_map=COLOR_MAP

    )

    fig.update_xaxes(

        title="Tahun"

    )

    fig.update_yaxes(

        title="Jumlah Review"

    )

    return apply_layout(

        fig,

        "Distribusi Sentimen per Tahun"

    )


# =========================================================
# HORIZONTAL BAR CHART
# =========================================================

def horizontal_bar(data, x, y, title, color=None):

    fig = px.bar(
        data,
        x=x,
        y=y,
        orientation="h",
        text=x,
        color=color if color else x,
        color_discrete_map=COLOR_MAP
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        yaxis=dict(categoryorder="total ascending")
    )

    return apply_layout(fig, title)


# =========================================================
# WORD FREQUENCY
# =========================================================

def word_frequency(word_series, title="Top 20 Kata"):

    data = word_series.reset_index()

    data.columns = [
        "Kata",
        "Frekuensi"
    ]

    fig = px.bar(
        data,
        x="Frekuensi",
        y="Kata",
        orientation="h",
        text="Frekuensi",
        color="Frekuensi",
        color_continuous_scale="Reds"
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    return apply_layout(
        fig,
        title
    )

# =========================================================
# BIGRAM CHART
# =========================================================

def bigram_chart(df, top_n=20):

    corpus = df["text"].fillna("").astype(str)

    vectorizer = CountVectorizer(
        ngram_range=(2,2),
        stop_words=None
    )

    X = vectorizer.fit_transform(corpus)

    words = vectorizer.get_feature_names_out()

    counts = X.sum(axis=0).A1

    result = pd.DataFrame({
        "Bigram": words,
        "Frekuensi": counts
    })

    result = result.sort_values(
        by="Frekuensi",
        ascending=False
    ).head(top_n)

    fig = px.bar(
        result,
        x="Frekuensi",
        y="Bigram",
        orientation="h",
        text="Frekuensi",
        color="Frekuensi",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    return apply_layout(
        fig,
        "Top Bigram"
    )

# =========================================================
# TRIGRAM CHART
# =========================================================

def trigram_chart(df, top_n=20):

    corpus = df["text"].fillna("").astype(str)

    vectorizer = CountVectorizer(
        ngram_range=(3,3),
        stop_words=None
    )

    X = vectorizer.fit_transform(corpus)

    words = vectorizer.get_feature_names_out()

    counts = X.sum(axis=0).A1

    result = pd.DataFrame({
        "Trigram": words,
        "Frekuensi": counts
    })

    result = result.sort_values(
        by="Frekuensi",
        ascending=False
    ).head(top_n)

    fig = px.bar(
        result,
        x="Frekuensi",
        y="Trigram",
        orientation="h",
        text="Frekuensi",
        color="Frekuensi",
        color_continuous_scale="Greens"
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    return apply_layout(
        fig,
        "Top Trigram"
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================

def confusion_matrix_chart(cm, labels):

    fig = px.imshow(

        cm,

        text_auto=True,

        x=labels,

        y=labels,

        color_continuous_scale="Blues"

    )

    fig.update_layout(

        xaxis_title="Predicted",

        yaxis_title="Actual"

    )

    return apply_layout(

        fig,

        "Confusion Matrix"

    )


# =========================================================
# STACKED SENTIMENT
# =========================================================

def stacked_sentiment(df):

    if "year" not in df.columns:
        return None

    yearly = (
        df.groupby(
            [
                "year",
                "pred_label"
            ]
        )
        .size()
        .reset_index(
            name="Jumlah"
        )
    )

    fig = px.bar(

        yearly,

        x="year",

        y="Jumlah",

        color="pred_label",

        barmode="relative",

        color_discrete_map=COLOR_MAP

    )

    return apply_layout(

        fig,

        "Stacked Sentiment"

    )


# =========================================================
# SENTIMENT TREND
# =========================================================

def sentiment_trend(df):

    if "year" not in df.columns:
        return None

    trend = (
        df.groupby(
            [
                "year",
                "pred_label"
            ]
        )
        .size()
        .reset_index(
            name="Jumlah"
        )
    )

    fig = px.line(

        trend,

        x="year",

        y="Jumlah",

        color="pred_label",

        markers=True,

        color_discrete_map=COLOR_MAP

    )

    fig.update_traces(
        line=dict(width=4)
    )

    return apply_layout(

        fig,

        "Trend Sentimen"

    )


# =========================================================
# HISTOGRAM RATING
# =========================================================

def rating_histogram(df):

    if "score" not in df.columns:
        return None

    fig = px.histogram(

        df,

        x="score",

        nbins=5,

        color="score",

        color_continuous_scale="Reds"

    )

    return apply_layout(

        fig,

        "Histogram Rating"

    )

# =========================================================
# ACCURACY GAUGE
# =========================================================

def accuracy_gauge(value):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#D71920"},
                "steps": [
                    {"range": [0, 60], "color": "#F8D7DA"},
                    {"range": [60, 80], "color": "#FFF3CD"},
                    {"range": [80, 100], "color": "#D4EDDA"},
                ],
            },
        )
    )

    return apply_layout(fig, "Accuracy")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

def classification_table(report_df):

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(report_df.columns),
                    fill_color="#D71920",
                    font=dict(color="white", size=13),
                    align="center",
                ),
                cells=dict(
                    values=[report_df[col] for col in report_df.columns],
                    align="center",
                ),
            )
        ]
    )

    fig.update_layout(height=420)

    return fig


# =========================================================
# RATING PIE
# =========================================================

def rating_pie(df):

    if "score" not in df.columns:
        return None

    rating = (
        df["score"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating.columns = ["Rating", "Jumlah"]

    fig = px.pie(
        rating,
        names="Rating",
        values="Jumlah",
        hole=.55
    )

    return apply_layout(
        fig,
        "Proporsi Rating"
    )


# =========================================================
# HEATMAP SENTIMENT
# =========================================================

def sentiment_heatmap(df):

    if "score" not in df.columns:
        return None

    heat = (
        df.groupby(
            [
                "score",
                "pred_label"
            ]
        )
        .size()
        .reset_index(name="Jumlah")
    )

    fig = px.density_heatmap(
        heat,
        x="score",
        y="pred_label",
        z="Jumlah",
        color_continuous_scale="Reds"
    )

    return apply_layout(
        fig,
        "Heatmap Rating vs Sentiment"
    )


# =========================================================
# PREDICTION PROBABILITY
# =========================================================

def probability_chart(labels, probs):

    fig = px.bar(

        x=labels,

        y=probs,

        color=labels,

        color_discrete_map=COLOR_MAP,

        text=[f"{i:.2%}" for i in probs]

    )

    fig.update_yaxes(

        title="Probability",

        range=[0, 1]

    )

    return apply_layout(

        fig,

        "Prediction Probability"

    )


# =========================================================
# EMPTY FIGURE
# =========================================================

def empty_chart(message="Data tidak tersedia"):

    fig = go.Figure()

    fig.add_annotation(

        text=message,

        showarrow=False,

        font=dict(size=18)

    )

    fig.update_xaxes(visible=False)

    fig.update_yaxes(visible=False)

    fig.update_layout(

        template="plotly_white",

        height=350

    )

    return fig


# =========================================================
# TOP SENTIMENT
# =========================================================

def top_sentiment(df):

    data = (
        df["pred_label"]
        .value_counts()
        .reset_index()
    )

    data.columns = [
        "Sentimen",
        "Jumlah"
    ]

    return data


# =========================================================
# EXPORT FIGURE
# =========================================================

def save_html(fig, filename):

    fig.write_html(filename)