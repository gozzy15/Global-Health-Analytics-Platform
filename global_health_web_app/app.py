"""
Main entry point for the Global Health Dashboard.
"""

from pathlib import Path

import streamlit as st

from utils.style import apply_global_styles

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Global Health Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()

# ---------------------------------------------------------
# Default dataset path
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_PATH = (
    BASE_DIR
    / "data"
    / "Global Health Dataset_cleaned.csv"
)


# ---------------------------------------------------------
# Check data availability
# ---------------------------------------------------------

default_dataset_exists = DEFAULT_DATA_PATH.exists()

uploaded_dataset_exists = (
    "uploaded_data" in st.session_state
    and st.session_state["uploaded_data"] is not None
    and not st.session_state["uploaded_data"].empty
)


data_available = (
    default_dataset_exists
    or uploaded_dataset_exists
)


# ---------------------------------------------------------
# Page definitions
# ---------------------------------------------------------

upload_page = st.Page(
    "pages/06_Data_Upload.py",
    title="Data Upload",
    icon="📁",
)


# ---------------------------------------------------------
# Navigation when data is available
# ---------------------------------------------------------

if data_available:

    overview_page = st.Page(
        "pages/01_Overview.py",
        title="Overview",
        icon="🌍",
    )

    country_analysis_page = st.Page(
        "pages/02_Country_Analysis.py",
        title="Country Analysis",
        icon="🌎",
    )

    disease_analysis_page = st.Page(
        "pages/03_Disease_Analysis.py",
        title="Disease Analysis",
        icon="🦠",
    )

    chi_analysis_page = st.Page(
        "pages/04_CHI_Analysis.py",
        title="CHI Analysis",
        icon="📊",
    )

    trends_analysis_page = st.Page(
        "pages/07_Trends_Analysis.py",
        title="Trends Analysis",
        icon="📈",
    )

    corr_analysis_page = st.Page(
        "pages/08_Correlation_Analysis.py",
        title="Correlation Analysis",
        icon="🔗",
    )

    stats_analysis_page = st.Page(
        "pages/09_Statistical_Analysis.py",
        title="Statistical Analysis",
        icon="📐",
    )

    machine_learning_page = st.Page(
        "pages/10_Machine_Learning.py",
        title="Machine Learning",
        icon="🤖",
    )

    data_explorer_page = st.Page(
        "pages/05_Data_Explorer.py",
        title="Data Explorer",
        icon="🔎",
    )

    reports_page = st.Page(
        "pages/11_Reports.py",
        title="Reports",
        icon="📄",
    )

    about_page = st.Page(
        "pages/12_About.py",
        title="About",
        icon="ℹ️",
    )

    pg = st.navigation(
        {
            "Analysis": [
                overview_page,
                country_analysis_page,
                disease_analysis_page,
                chi_analysis_page,
                trends_analysis_page,
                corr_analysis_page,
                stats_analysis_page,
                machine_learning_page,
                data_explorer_page,
                reports_page,
                about_page,
            ],
            "Data": [
                upload_page,
            ],
        }
    )

else:

    # -----------------------------------------------------
    # No dataset available
    # -----------------------------------------------------

    pg = st.navigation(
        {
            "Data": [
                upload_page,
            ],
        }
    )


# ---------------------------------------------------------
# Sidebar footer
# ---------------------------------------------------------

st.sidebar.markdown(
    """
    <div style="
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128, 128, 128, 0.15);
        text-align: center;
        color: rgba(128, 128, 128, 1.00);
        font-size: 0.85rem;
        line-height: 1.5;
    ">
        <div>Built by <strong>Chigozie Nnoli</strong></div>
        <div>Data Analyst &amp; BI Professional</div>
        <div style="margin-top: 0.25rem;">
            <a href="https://github.com/gozzy15/"
               target="_blank"
               style="color: inherit; text-decoration: none;">
                GitHub
            </a>
            &nbsp;•&nbsp;
            <a href="https://gozzydanalyst.my.canva.site"
               target="_blank"
               style="color: inherit; text-decoration: none;">
                Portfolio
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Run selected page
# ---------------------------------------------------------

pg.run()