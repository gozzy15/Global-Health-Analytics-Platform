"""
Reusable helper functions for the Global Health Dashboard.
"""

from pathlib import Path

import streamlit as st


def load_css(css_path: str | Path) -> None:
    """
    Load and inject a CSS stylesheet into the Streamlit app.

    Parameters
    ----------
    css_path : str or Path
        Path to the CSS file.
    """
    path = Path(css_path)

    if not path.exists():
        st.warning(f"CSS file not found: {path}")
        return

    css = path.read_text(encoding="utf-8")

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def format_number(value: float | int, decimals: int = 0) -> str:
    """
    Format a numeric value with thousands separators.

    Examples
    --------
    10000 -> '10,000'
    1234.56 -> '1,234.56'
    """
    if value is None:
        return "N/A"

    try:
        return f"{value:,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_percentage(value: float | int, decimals: int = 1) -> str:
    """
    Format a numeric value as a percentage.

    Parameters
    ----------
    value : float or int
        Percentage value.
    decimals : int
        Number of decimal places.
    """
    if value is None:
        return "N/A"

    try:
        return f"{value:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_currency(
    value: float | int,
    decimals: int = 0,
    currency_symbol: str = "$",
) -> str:
    """
    Format a numeric value as currency.
    """
    if value is None:
        return "N/A"

    try:
        return f"{currency_symbol}{value:,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def display_page_header(
    title: str,
    subtitle: str | None = None,
) -> None:
    """
    Display a consistent dashboard page header.
    """
    st.markdown(
        f'<div class="dashboard-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(
            f'<div class="dashboard-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def display_section_title(title: str) -> None:
    """
    Display a consistent section heading.
    """
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )