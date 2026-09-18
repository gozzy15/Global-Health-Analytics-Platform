"""
Global styling utilities for the Global Health Dashboard.
"""

import streamlit as st


def apply_global_styles() -> None:
    """
    Apply consistent, theme-aware styling across the dashboard.
    """

    st.markdown(
        """
        <style>

        /* -------------------------------------------------
           Main application
        ------------------------------------------------- */

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }


        /* -------------------------------------------------
           Page titles
        ------------------------------------------------- */

        h1 {
            font-weight: 700;
            letter-spacing: -0.5px;
        }


        /* -------------------------------------------------
           Section headings
        ------------------------------------------------- */

        h2 {
            font-weight: 650;
            margin-top: 1.5rem;
        }

        h3 {
            font-weight: 600;
        }


        /* -------------------------------------------------
           Metric cards
        ------------------------------------------------- */

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
            padding: 1rem;
        }

        div[data-testid="stMetricLabel"] {
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 700;
        }

        div[data-testid="stMetricDelta"] {
            font-weight: 600;
        }


        /* -------------------------------------------------
           Buttons
        ------------------------------------------------- */

        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            min-height: 2.5rem;
        }


        /* -------------------------------------------------
           Download buttons
        ------------------------------------------------- */

        .stDownloadButton > button {
            border-radius: 8px;
            font-weight: 600;
            min-height: 2.5rem;
        }


        /* -------------------------------------------------
           Alerts and information boxes
        ------------------------------------------------- */

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }


        /* -------------------------------------------------
           Dataframes
        ------------------------------------------------- */

        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }


        /* -------------------------------------------------
           Expanders
        ------------------------------------------------- */

        div[data-testid="stExpander"] {
            border-radius: 8px;
        }


        /* -------------------------------------------------
           Sidebar
        ------------------------------------------------- */

        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.25);
        }


        /* -------------------------------------------------
           Select boxes and multiselects
        ------------------------------------------------- */

        div[data-baseweb="select"] > div {
            border-radius: 8px;
        }


        /* -------------------------------------------------
           File uploader
        ------------------------------------------------- */

        div[data-testid="stFileUploader"] {
            border-radius: 8px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )