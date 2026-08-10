"""
Cards.py Nazalyst
"""

import streamlit as st


# ==========================================================
# INTERNAL RENDER
# ==========================================================

def _render(html: str):
    """
    Render HTML tanpa indentasi agar tidak diparse sebagai
    Markdown code block.
    """
    st.markdown(html.strip(), unsafe_allow_html=True)


# ==========================================================
# SECTION TITLE
# ==========================================================

def section_title(title: str):

    _render(
        f'<div class="section-title">{title}</div>'
    )


# ==========================================================
# KPI CARD
# ==========================================================

def metric_card(
    title: str,
    value,
    icon: str = "📊",
    color: str = "#D71920",
):

    html = (
        f'<div class="metric-card" '
        f'style="border-left:6px solid {color};">'
        f'<div class="metric-icon">{icon}</div>'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>'
    )

    _render(html)


# ==========================================================
# INFO CARD
# ==========================================================

def info_card(
    title: str,
    content: str,
    color: str = "#FFFFFF",
):

    html = (
        f'<div class="card" '
        f'style="background:{color};">'
        f'<h4>{title}</h4>'
        f'<div class="card-content">{content}</div>'
        f'</div>'
    )

    _render(html)


# ==========================================================
# SUCCESS CARD
# ==========================================================

def success_card(
    title: str,
    content: str,
):

    html = (
        '<div class="card" '
        'style="border-left:8px solid #16A34A;">'
        f'<h3>✅ {title}</h3>'
        f'<div class="card-content">{content}</div>'
        '</div>'
    )

    _render(html)


# ==========================================================
# WARNING CARD
# ==========================================================

def warning_card(
    title: str,
    content: str,
):

    html = (
        '<div class="card" '
        'style="border-left:8px solid #F59E0B;">'
        f'<h3>⚠️ {title}</h3>'
        f'<div class="card-content">{content}</div>'
        '</div>'
    )

    _render(html)


# ==========================================================
# ERROR CARD
# ==========================================================

def error_card(
    title: str,
    content: str,
):

    html = (
        '<div class="card" '
        'style="border-left:8px solid #DC2626;">'
        f'<h3>❌ {title}</h3>'
        f'<div class="card-content">{content}</div>'
        '</div>'
    )

    _render(html)


# ==========================================================
# ABOUT CARD
# ==========================================================

def about_card(
    title: str,
    body: str,
):

    html = (
        '<div class="card">'
        f'<h3>{title}</h3>'
        '<hr>'
        f'<div class="card-content" '
        'style="text-align:justify;">'
        f'{body}'
        '</div>'
        '</div>'
    )

    _render(html)


# ==========================================================
# STATUS BADGE
# ==========================================================

def status_badge(
    text: str,
    color: str = "#16A34A",
):

    _render(
        f'<span style="'
        f'background:{color};'
        f'color:white;'
        f'padding:4px 12px;'
        f'border-radius:999px;'
        f'font-size:13px;'
        f'font-weight:600;">'
        f'{text}'
        f'</span>'
    )