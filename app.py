import base64

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Fraudlens | Fraud Risk Intelligence",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent

LOGO_PATH = APP_DIR / "fraudlens_logo.svg"
LOGO_URI = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    if LOGO_PATH.exists()
    else ""
)

MODEL_PATH = APP_DIR / "fraud_detection_model_v2.joblib"
SAMPLE_PATH = APP_DIR / "sample_transactions.csv"

# --------------------------------------------------
# PROFESSIONAL DARK THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #07111F 0%, #0B1728 55%, #101F32 100%);
    color: #E8F0FA;
}
[data-testid="stSidebar"] {
    background: #0A1423;
    border-right: 1px solid #24364B;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
.hero {
    padding: 30px;
    border: 1px solid #29445A;
    border-radius: 22px;
    background: linear-gradient(120deg, #10263B, #102D35);
    margin-bottom: 24px;
}
.eyebrow {
    color: #4DE0C1;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.hero h1 {
    color: #F4F8FF;
    font-size: 38px;
    margin: 10px 0;
    line-height: 1.2;
}
.hero p {
    color: #B5C7D9;
    font-size: 15px;
    margin-bottom: 0;
}
.section-title {
    color: #F4F8FF;
    font-size: 21px;
    font-weight: 700;
    margin-top: 12px;
}
.small-muted {
    color: #91A6BB;
    font-size: 13px;
}
div[data-testid="stMetric"] {
    background: #101F31;
    border: 1px solid #263D52;
    padding: 18px;
    border-radius: 15px;
}
div[data-testid="stMetricLabel"] {
    color: #A9BED1;
}
div[data-testid="stMetricValue"] {
    color: #F2F8FF;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    border: 1px solid #3BCDB5;
    min-height: 44px;
    font-weight: 600;
}
.stButton > button[kind="primary"] {
    background: #21B99E;
    color: #061821;
}
div[data-testid="stFileUploader"] {
    background: #101F31;
    border: 1px dashed #3A586D;
    padding: 15px;
    border-radius: 14px;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #263D52;
    border-radius: 12px;
}
hr {
    border-color: #263D52;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
@st.cache_resource
def load_bundle(model_path):
    return joblib.load(model_path)

try:
    bundle = load_bundle(str(MODEL_PATH))
    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = float(bundle["threshold"])
    feature_columns = list(bundle["feature_columns"])
except Exception:
    st.error(
        "We couldn't load the saved model. Please check that "
        "the model file is present in the application folder."
    )
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">INTELLIGENCE • ANALYTICS • SECURITY</div>
    {
        f'<img src="{LOGO_URI}" alt="FRAUDLENS logo" '
        'style="width:100%;max-width:440px;height:auto;'
        'display:block;margin:12px auto 18px;">'
        if LOGO_URI else '<h1>FRAUDLENS</h1>'
    }
    <p><b>Fraud Risk Intelligence Platform</b></p>
    <p>Explore transaction patterns, estimate fraud risk, and turn
       model predictions into clear, actionable insights.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🔐 Fraudlens")
    st.caption("Fraud Risk Intelligence")
    st.divider()

    st.markdown("### ⚙️ Model Information")
    st.write("**Algorithm**")
    st.write(bundle.get("model_name", "Saved ML model"))

    st.write("**Decision threshold**")
    st.code(f"{threshold:.4f}")

    st.write("**Input features**")
    st.write(f"{len(feature_columns)} required features")

    st.divider()
    st.markdown("### 📘 About this project")
    st.write(
        "An academic machine-learning project that estimates "
        "transaction fraud risk using anonymized transaction features."
    )

    st.caption(
        "Demo only. Predictions are estimates, not confirmed fraud."
    )

# --------------------------------------------------
# TOP OVERVIEW
# --------------------------------------------------
st.markdown('<div class="section-title">📊 Dashboard Overview</div>',
            unsafe_allow_html=True)
st.caption("Upload transaction data to begin your analysis.")

overview1, overview2, overview3 = st.columns(3)

overview1.metric("Model", bundle.get("model_name", "ML model"))
overview2.metric("Risk threshold", f"{threshold:.2%}")
overview3.metric("Required features", len(feature_columns))

st.divider()

# --------------------------------------------------
# DATA INPUT
# --------------------------------------------------
st.markdown('<div class="section-title">📂 Transaction Workspace</div>',
            unsafe_allow_html=True)

upload_col, sample_col = st.columns([1.5, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload transaction data",
        type=["csv"],
        help="Use the original compatible format: Time, V1–V28, Amount."
    )

with sample_col:
    st.markdown("#### 🧪 Explore the demo")
    st.write(
        "New to the dashboard? Use the sample dataset to explore "
        "the results without uploading your own file."
    )
    sample_clicked = st.button(
        "✨ Try Sample Transactions",
        type="primary",
        use_container_width=True
    )

data = None

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.info("📄 Your uploaded dataset is selected.")
    except Exception:
        st.error(
            "We couldn't read this file. Please upload a valid CSV file."
        )
        st.stop()

elif sample_clicked:
    try:
        data = pd.read_csv(SAMPLE_PATH)
        st.success("🌟 Sample dataset loaded! Let's explore the results.")
    except Exception:
        st.error(
            "The sample dataset could not be loaded. "
            "Check that sample_transactions.csv is in the app folder."
        )
        st.stop()

# --------------------------------------------------
# VALIDATION AND PREDICTION
# --------------------------------------------------
if data is not None:
    st.divider()
    st.markdown(
        '<div class="section-title">🔍 Analysis Results</div>',
        unsafe_allow_html=True
    )

    if data.empty:
        st.error(
            "This file has no transaction rows yet. "
            "Add data and upload it again."
        )
        st.stop()

    if data.columns.duplicated().any():
        st.error(
            "Some column names are duplicated. Please correct the CSV "
            "so every column has a unique name."
        )
        st.stop()

    missing = sorted(set(feature_columns) - set(data.columns))

    if missing:
        st.error("Some required model features are missing.")
        st.write("Missing features:", ", ".join(missing))
        st.info(
            "This model requires the original Time, V1–V28 and Amount "
            "feature format. An unrelated dataset cannot be made "
            "compatible simply by renaming its columns."
        )
        st.stop()

    if len(data) > 100000:
        st.error(
            "This demo supports up to 100,000 rows per upload. "
            "Please split your file into smaller CSVs."
        )
        st.stop()

    X_input = data[feature_columns].copy()

    for col in feature_columns:
        X_input[col] = pd.to_numeric(X_input[col], errors="coerce")

    if X_input.isna().any().any():
        bad_cols = X_input.columns[
            X_input.isna().any()
        ].tolist()
        st.error(
            "Missing or non-numeric values were found in: "
            + ", ".join(bad_cols)
        )
        st.info("Correct those values and upload the CSV again.")
        st.stop()

    if not np.isfinite(X_input.to_numpy(dtype=float)).all():
        st.error(
            "The dataset contains infinite or invalid numeric values. "
            "Please clean the affected cells and try again."
        )
        st.stop()

    st.success(
        f"✅ Data checks passed — {len(X_input):,} transactions are ready."
    )

    try:
        if "Time" in feature_columns and "Amount" in feature_columns:
            X_input[["Time", "Amount"]] = scaler.transform(
                X_input[["Time", "Amount"]]
            )

        probabilities = model.predict_proba(X_input)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        results = data.copy()
        results["Fraud Probability (%)"] = (
            probabilities * 100
        ).round(2)
        results["Predicted Fraud"] = predictions
        results["Risk Level"] = np.where(
            predictions == 1, "High Risk", "Low Risk"
        )

        total = len(results)
        flagged = int(predictions.sum())
        low_risk = total - flagged
        avg_probability = float(probabilities.mean() * 100)

        # Summary cards
        st.markdown("#### 📌 Risk Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transactions analyzed", f"{total:,}")
        m2.metric("⚠️ High-risk flags", f"{flagged:,}")
        m3.metric("✓ Low-risk predictions", f"{low_risk:,}")
        m4.metric("Average estimated probability", f"{avg_probability:.2f}%")

        # Friendly result messages
        if flagged == 0:
            st.success(
                "✨ All clear for this batch! The model did not flag any "
                "transactions at the current threshold. This does not "
                "guarantee that every transaction is legitimate."
            )
        elif flagged == total:
            st.warning(
                "🔔 Every transaction in this batch was flagged as high "
                "risk. Review the data and model outputs carefully; "
                "a flag is not proof of fraud."
            )
        else:
            st.warning(
                f"🔎 The model flagged {flagged:,} of {total:,} "
                "transactions for further review. Check the results "
                "before drawing conclusions."
            )

        # Visual analytics
        st.markdown("#### 📈 Visual Insights")
        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown("**Risk distribution**")
            risk_counts = pd.DataFrame({
                "Transactions": [low_risk, flagged]
            }, index=["Low Risk", "High Risk"])
            st.bar_chart(risk_counts)

        with chart2:
            st.markdown("**Fraud probability by transaction**")
            chart_data = pd.DataFrame({
                "Estimated probability (%)": (
                    probabilities[:100] * 100
                )
            })
            chart_data.index = range(1, len(chart_data) + 1)
            st.line_chart(chart_data)
            if total > 100:
                st.caption("Chart displays the first 100 transactions.")

        # Results table
        st.markdown("#### 🧾 Transaction-Level Results")
        risk_filter = st.selectbox(
            "Filter by risk level",
            ["All transactions", "High Risk", "Low Risk"]
        )

        if risk_filter == "High Risk":
            display_results = results[
                results["Risk Level"] == "High Risk"
            ]
        elif risk_filter == "Low Risk":
            display_results = results[
                results["Risk Level"] == "Low Risk"
            ]
        else:
            display_results = results

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Analysis Results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="sentinel_fraud_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "Risk labels are based on the saved model's decision "
            "threshold. Model performance depends on the training data "
            "and should be validated against verified labels."
        )

    except Exception:
        st.error(
            "The analysis could not be completed. Please verify that "
            "your dataset matches the model's expected feature format."
        )

else:
    st.info(
        "👋 Welcome! Upload a compatible CSV or select "
        "'Try Sample Transactions' to see the dashboard in action."
    )

    st.markdown("#### 🚀 Your workflow")
    step1, step2, step3 = st.columns(3)

    step1.markdown("### 01")
    step1.write("**Upload**")
    step1.caption("Provide a compatible transaction CSV.")

    step2.markdown("### 02")
    step2.write("**Analyze**")
    step2.caption("Estimate transaction risk with the saved model.")

    step3.markdown("### 03")
    step3.write("**Review**")
    step3.caption("Explore charts and download the results.")

st.divider()
st.markdown(
    '<p class="small-muted" style="text-align:center;">'
    'FRAUDLENS • FRAUD RISK INTELLIGENCE<br>'
    'Academic portfolio project | Not for production banking decisions'
    '</p>',
    unsafe_allow_html=True
)
