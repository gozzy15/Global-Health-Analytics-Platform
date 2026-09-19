"""
About page for the Global Health Dashboard.
"""

import streamlit as st

from utils.style import apply_global_styles


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="About | Global Health Dashboard",
    page_icon="ℹ️",
    layout="wide",
)

apply_global_styles()


# =========================================================
# PAGE HEADER
# =========================================================

st.title("ℹ️ About the Global Health Dashboard")

st.markdown(
    """
    The **Global Health Dashboard** is an analytical application
    designed to explore, understand, and model patterns in global
    health data.

    The application brings together data exploration, statistical
    analysis, Composite Health Index analysis, machine learning,
    predictive modeling, and automated reporting within a single
    interactive environment.
    """
)


# =========================================================
# ABOUT TABS
# =========================================================

(
    tab_overview,
    tab_data_analytics,
    tab_reporting_technology,
    tab_usage,
    tab_scope_project,
    tab_author,
) = st.tabs(
    [
        "📖 Overview",
        "📊 Data & Analytics",
        "📤 Reporting & Technology",
        "📘 Usage Guide",
        "🔒 Scope & Project",
        "👤 Author",
    ]
)


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with tab_overview:

    st.header("About the Application")

    st.markdown(
        """
        The **Global Health Dashboard** provides a structured
        environment for transforming health data into meaningful
        analytical insights.

        It is designed to support a complete data analytics
        workflow, from data preparation and exploration through
        statistical analysis, visualization, machine learning,
        prediction, and reporting.

        Key capabilities include:

        - Data preparation and validation
        - Exploratory data analysis
        - Statistical analysis
        - Health trend analysis
        - Composite Health Index analysis
        - Machine learning
        - Predictive modeling
        - Country-level analysis
        - Data exploration and filtering
        - Data and analytical exports
        - Automated report generation
        - Email-based report delivery

        The application demonstrates how data engineering,
        statistical analysis, visualization, business intelligence,
        and machine learning can be combined into a practical
        data-driven analytical system.
        """
    )

    st.divider()

    st.header("Purpose")

    st.markdown(
        """
        The primary purpose of the dashboard is to make complex
        health data easier to explore, analyze, and interpret.

        Rather than relying on isolated analyses or static tables,
        the application provides an interactive environment where
        users can examine health indicators across countries,
        diseases, and years.

        The dashboard also provides analytical and predictive tools
        for investigating relationships between healthcare,
        socioeconomic conditions, disease indicators, and broader
        health outcomes.

        The application is intended primarily for analytical,
        educational, and demonstration purposes.
        """
    )


# =========================================================
# TAB 2 — DATA & ANALYTICS
# =========================================================

with tab_data_analytics:

    st.header("Dataset Overview")

    st.markdown(
        """
        The application uses the **Global Health Dataset (2000–2024)**.

        The dataset contains information covering:

        - **20 countries**
        - **20 diseases**
        - **25 years**
        - **10,000 records**
        - Health, healthcare, demographic, and socioeconomic
          indicators

        Important indicators include:

        - Incidence Rate
        - Prevalence Rate
        - Mortality Rate
        - Healthcare Access
        - Doctors per 1,000
        - Hospital Beds per 1,000
        - Recovery Rate
        - DALYs
        - Composite Health Index
        - Per Capita Income
        - Education Index
        - Urbanization Rate

        The dataset was cleaned and validated before being used
        throughout the analytical and machine-learning stages.
        """
    )

    st.divider()

    st.header("Analytical Capabilities")

    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:

        st.subheader("📊 Statistical Analysis")

        st.markdown(
            """
            The dashboard provides descriptive and statistical
            analysis of health indicators, including:

            - Descriptive statistics
            - Yearly statistics
            - Country-level statistics
            - Outlier analysis
            - Distribution analysis
            - Trend analysis
            - Statistical relationships between indicators
            """
        )

    with analysis_col2:

        st.subheader("📈 CHI Analysis")

        st.markdown(
            """
            The Composite Health Index analysis examines:

            - CHI trends
            - Country-level CHI patterns
            - Relationships with health indicators
            - Correlation analysis
            - Regression analysis
            - Multicollinearity diagnostics
            - Feature importance
            - Country clustering
            """
        )

    st.divider()

    st.header("Machine Learning")

    st.markdown(
        """
        The machine-learning component investigates whether
        historical health, healthcare, demographic, and
        socioeconomic information can be used to model and predict
        selected health outcomes.

        Supported prediction targets include:

        - **Incidence Rate (%)**
        - **Mortality Rate (%)**
        - **Recovery Rate (%)**

        The machine-learning workflow includes:

        - Data preparation
        - Time-based train/test splitting
        - Model training
        - Model evaluation
        - Historical and future predictions
        - Manual prediction scenarios
        - Feature importance
        - Country clustering
        - Model evaluation exports
        - Prediction exports
        - Feature-importance exports
        - Country-clustering exports

        Prediction results are analytical model outputs and should
        not be interpreted as guaranteed future outcomes.
        """
    )


# =========================================================
# TAB 3 — REPORTING & TECHNOLOGY
# =========================================================

with tab_reporting_technology:

    st.header("Reporting & Exports")

    st.markdown(
        """
        The dashboard provides multiple ways to preserve,
        communicate, and reuse analytical results.

        Available outputs include:

        ### Dataset Exports

        - Filtered datasets in CSV format
        - Filtered datasets in Excel format

        ### Analysis Exports

        - Trend analysis
        - Correlation analysis
        - Statistical analysis
        - Composite Health Index analysis

        ### Machine Learning Exports

        - Model evaluation
        - Predictions
        - Feature importance
        - Country clustering

        ### Reports

        - PDF reports with charts
        - Interactive HTML reports
        - Email delivery of generated reports

        Generated reports are designed to provide convenient
        analytical summaries that can be reviewed outside the
        dashboard.
        """
    )

    st.divider()

    st.header("Technology Stack")

    st.markdown(
        """
        The Global Health Dashboard is built using Python-based
        technologies for data processing, statistical analysis,
        machine learning, visualization, reporting, and web
        application development.
        """
    )

    technology_col1, technology_col2 = st.columns(2)

    with technology_col1:

        st.subheader("🐍 Core Development & Data")

        st.markdown(
            """
            **Python**  
            Primary programming language used throughout the
            application.

            **Pandas**  
            Used for data loading, cleaning, transformation,
            filtering, aggregation, and analysis.

            **NumPy**  
            Used for numerical operations and data processing.

            **Scikit-learn**  
            Used for machine-learning models, prediction,
            feature importance, and country clustering.

            **Statsmodels**  
            Used for statistical modelling, regression analysis,
            significance testing, and related diagnostics.
            """
        )

    with technology_col2:

        st.subheader("📊 Application, Visualization & Reporting")

        st.markdown(
            """
            **Streamlit**  
            Provides the interactive web application and dashboard
            interface.

            **Plotly**  
            Used to create interactive analytical visualizations
            and charts.

            **ReportLab**  
            Used to generate PDF reports programmatically.

            **Kaleido**  
            Used to convert Plotly visualizations into images for
            inclusion in PDF reports.

            **OpenPyXL**  
            Used to generate Excel-based analytical exports.

            **python-dotenv**  
            Used for environment-based application configuration,
            including email-reporting settings.
            """
        )

    st.divider()

    st.header("Technology Architecture")

    st.markdown(
        """
        The application follows a modular architecture in which
        data loading, calculations, analysis, machine learning,
        visualization, reporting, and email functionality are
        separated into reusable components.

        This structure helps individual parts of the system remain
        maintainable, testable, and extensible without unnecessarily
        coupling the entire application together.

        The architecture also supports caching and reusable
        analytical functions to improve application performance
        during repeated interactions.
        """
    )


# =========================================================
# TAB 4 — USAGE GUIDE
# =========================================================

with tab_usage:

    st.header("Usage Instructions")

    st.markdown(
        """
        The Global Health Dashboard is organized into several
        sections, with each section serving a specific analytical
        purpose.
        """
    )

    st.subheader("1. Home")

    st.markdown(
        """
        Start from the **Home** page to access the dashboard and
        understand its main capabilities.
        """
    )

    st.subheader("2. Data Upload")

    st.markdown(
        """
        Use **Data Upload** when you need to provide a dataset
        for analysis.

        Before uploading a file, ensure that the data is properly
        structured and does not contain confidential, personally
        identifiable, or sensitive information that you are not
        authorized to process.
        """
    )

    st.subheader("3. Data Analysis")

    st.markdown(
        """
        Use **Data Analysis** to examine the dataset and explore
        key health indicators through analytical summaries and
        visualizations.
        """
    )

    st.subheader("4. Data Explorer")

    st.markdown(
        """
        Use **Data Explorer** when you want to interactively
        filter the dataset.

        You can explore available records by selecting countries,
        diseases, years, and other available filters.

        Filtered datasets can also be exported in CSV and Excel
        formats.
        """
    )

    st.subheader("5. Trends")

    st.markdown(
        """
        Use **Trends** to examine how health indicators change
        over time.

        The section allows you to investigate changes across
        countries and years and export the resulting trend
        analysis.
        """
    )

    st.subheader("6. Statistical Analysis")

    st.markdown(
        """
        Use **Statistical Analysis** to examine the statistical
        characteristics of selected health indicators.

        Available analyses include descriptive statistics,
        yearly statistics, country-level statistics, distribution
        analysis, and outlier analysis.
        """
    )

    st.subheader("7. CHI Analysis")

    st.markdown(
        """
        Use **CHI Analysis** to investigate the Composite Health
        Index and its relationship with other health, healthcare,
        demographic, and socioeconomic indicators.

        The section includes correlation analysis, regression
        analysis, multicollinearity diagnostics, feature
        importance, and country clustering.
        """
    )

    st.subheader("8. Machine Learning")

    st.markdown(
        """
        Use **Machine Learning** to prepare data, train models,
        evaluate model performance, generate predictions, examine
        feature importance, and explore country-level clustering.

        Prediction results should be interpreted as analytical
        outputs rather than guaranteed future outcomes.
        """
    )

    st.subheader("9. Reports")

    st.markdown(
        """
        Use **Reports** to generate and preserve analytical
        results.

        Available report options include:

        - PDF reports with charts
        - Interactive HTML reports
        - Email delivery of generated reports

        The Reports page also provides access to generated
        analytical outputs and report-related functionality.
        """
    )

    st.subheader("10. Exports")

    st.markdown(
        """
        Where available, use the export controls within the
        relevant sections to download analytical results.

        Export formats include CSV and Excel, depending on the
        analysis being performed.
        """
    )

    st.divider()

    st.header("Recommended Workflow")

    st.markdown(
        """
        **Explore → Filter → Analyze → Visualize → Model → Export → Report**

        Start by exploring the data, investigate trends and
        statistical relationships, examine the Composite Health
        Index, perform machine-learning analysis where appropriate,
        and then export or report the results.
        """
    )


# =========================================================
# TAB 5 — SCOPE, PRIVACY & PROJECT
# =========================================================

with tab_scope_project:

    st.header("Project Scope & Limitations")

    st.markdown(
        """
        This application is an analytical and educational data
        science project built around a **synthetic Global Health
        Dataset (2000–2024)**.

        The dataset is artificially generated for analytical,
        visualization, statistical, and machine-learning purposes.
        It is **not an official public-health dataset** and should
        not be interpreted as representing verified measurements
        of actual countries, diseases, populations, or health
        outcomes.

        Consequently, the values and patterns presented in the
        dashboard should not be used as a substitute for official
        public-health surveillance, clinical guidance, government
        statistics, or medical decision-making.

        Machine-learning predictions and statistical relationships
        should also be interpreted as analytical results rather than
        definitive causal conclusions.

        In particular, statistical association does not by itself
        establish causation.
        """
    )

    st.divider()

    st.header("Privacy Information")

    st.markdown(
        """
        The Global Health Dashboard is designed with privacy and
        responsible data handling in mind.

        ### Data Used by the Application

        The primary dataset used by this application is a
        **synthetic Global Health Dataset (2000–2024)**. It does not
        represent verified personal health records or identifiable
        patient information.

        The dataset is used for analytical, statistical,
        visualization, and machine-learning purposes.

        ### Personal Information

        The dashboard does not require users to provide personal
        health information in order to explore the dataset or use
        its analytical features.

        Users should not upload personally identifiable information,
        confidential medical records, patient records, or other
        sensitive personal information into the application.

        ### Uploaded Data

        If the application provides data-upload functionality,
        users are responsible for ensuring that uploaded files do
        not contain information that they are not authorized to
        process or share.

        Uploaded datasets should be limited to information necessary
        for the intended analytical purpose.

        ### Report and Export Data

        Reports and exported files may contain analytical results,
        filtered datasets, statistical summaries, predictions, and
        other information generated from the data provided to the
        application.

        Users should handle downloaded reports and exported files
        appropriately, particularly if the application is later
        used with non-public datasets.

        ### Email Reports

        When the email-reporting feature is used, the application
        sends the selected report to the email address provided by
        the user.

        Users should ensure that reports are sent only to intended
        recipients and that the contents of the report are
        appropriate for transmission by email.

        ### Responsible Data Use

        Privacy, security, and responsible data handling should be
        considered whenever the application is extended to work
        with real-world or confidential datasets.

        The current synthetic dataset is intended for analytical
        and educational purposes and does not constitute a database
        of real patient or personal health information.
        """
    )

    st.divider()

    st.header("Project Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.markdown(
            """
            **Project:** Global Health Dashboard

            **Dataset:** Global Health Dataset (2000–2024)

            **Application Type:** Interactive Data Analytics &
            Machine Learning Dashboard
            """
        )

    with info_col2:

        st.markdown(
            """
            **Primary Focus:**

            Global health analytics, statistical analysis,
            predictive modeling, visualization, and automated
            reporting.

            **Development Approach:**

            Modular, reusable, analytical, and
            workflow-oriented.
            """
        )


# =========================================================
# TAB 6 — AUTHOR
# =========================================================

with tab_author:

    st.header("About the Author")

    st.markdown(
        """
        **Chigozie Nnoli** is a **Data Analyst and Business
        Intelligence professional** with a background in **Physics
        with Electronics** and professional training in **Data
        Analytics, Data Science, Machine Learning, and Python**
        through **ALX Africa**.

        His work focuses on transforming raw and complex datasets
        into meaningful insights through data cleaning, exploratory
        analysis, statistical analysis, data visualization,
        business intelligence, machine learning, and analytical
        automation.

        The Global Health Dashboard reflects this approach by
        combining data preparation, analytical exploration,
        statistical modelling, machine learning, visualization,
        and reporting into a single end-to-end application.
        """
    )

    st.divider()

    st.header("Professional Focus")

    focus_col1, focus_col2 = st.columns(2)

    with focus_col1:

        st.markdown(
            """
            **Core Areas**

            - Data Analytics
            - Business Intelligence
            - SQL
            - Python
            - Excel
            - Power BI
            - Data Cleaning & Transformation
            - Exploratory Data Analysis
            """
        )

    with focus_col2:

        st.markdown(
            """
            **Advanced Areas**

            - Statistical Analysis
            - Data Visualization
            - Machine Learning
            - Predictive Modelling
            - Feature Analysis
            - Data Automation
            - Analytical Reporting
            - AI-Powered Analytics
            """
        )

    st.divider()

    st.header("Connect")

    st.markdown(
        """
        Explore more of Chigozie's work, projects, and professional
        background through the links below.
        """
    )

    link_col1, link_col2, link_col3 = st.columns(3)

    with link_col1:

        st.markdown(
            """
            **GitHub**

            [View Projects](https://github.com/gozzy15/)
            """
        )

    with link_col2:

        st.markdown(
            """
            **Portfolio**

            [View Portfolio](https://gozzydanalyst.my.canva.site)
            """
        )

    with link_col3:

        st.markdown(
            """
            **LinkedIn**

            [Connect on LinkedIn](https://www.linkedin.com/in/chigozie-nnoli)
            """
        )

    st.divider()

    st.caption(
        "This project was developed as part of an end-to-end "
        "data analytics and machine-learning portfolio."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Global Health Dashboard • Data Analytics, Statistical "
    "Analysis, Machine Learning & Reporting"
)