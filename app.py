
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="FRAUDLENS | Fraud Risk Intelligence",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGO_PATH = APP_DIR / "fraudlens_logo.svg"
MODEL_PATH = APP_DIR / "fraud_detection_model_v2.joblib"
SAMPLE_PATH = APP_DIR / "sample_transactions.csv"

# --------------------------------------------------
# PROFESSIONAL DARK THEME
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        145deg, #07111F 0%, #0B1728 55%, #101F32 100%
    );
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
# LOAD SAVED MODEL
# --------------------------------------------------

@st.cache_resource
def load_bundle(model_path):
    return joblib.load(model_path)


try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    bundle = load_bundle(str(MODEL_PATH))

    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = float(bundle["threshold"])
    feature_columns = list(bundle["feature_columns"])

    if not feature_columns:
        raise ValueError("The saved model has no feature columns.")

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("The model contains duplicate feature names.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The saved decision threshold is invalid.")

    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError(
            "The saved model does not support predict_proba()."
        )

except Exception as exc:
    st.error(
        "The saved model could not be loaded. Check that "
        "`fraud_detection_model_v2.joblib` exists and contains "
        "the expected model, scaler, threshold and feature columns."
    )
    st.caption(f"Technical detail: {exc}")
    st.stop()

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def read_csv_safely(file):
    """Read a CSV and report common file-format problems."""

    try:
        df = pd.read_csv(file, encoding="utf-8-sig")

    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "The CSV is empty or does not contain a header row."
        ) from exc

    except pd.errors.ParserError as exc:
        raise ValueError(
            "The CSV format is invalid. Check quotation marks, "
            "separators and rows with inconsistent column counts."
        ) from exc

    except UnicodeDecodeError as exc:
        raise ValueError(
            "The file encoding could not be read. Save it as a "
            "UTF-8 CSV and upload it again."
        ) from exc

    except Exception as exc:
        raise ValueError(
            f"The file could not be read ({type(exc).__name__})."
        ) from exc

    # Normalize accidental spaces in headers.
    df.columns = [
        col.strip() if isinstance(col, str) else col
        for col in df.columns
    ]

    if df.columns.duplicated().any():
        duplicates = df.columns[
            df.columns.duplicated()
        ].tolist()

        raise ValueError(
            "Duplicate column names were found: "
            + ", ".join(map(str, duplicates))
            + ". Rename duplicate columns and upload again."
        )

    if df.empty:
        raise ValueError(
            "The CSV contains headers but no transaction rows."
        )

    return df


def validate_features(df):
    """Validate model features without changing their meanings."""

    if len(df) > 100000:
        raise ValueError(
            "This demo supports up to 100,000 rows per upload. "
            "Split the CSV into smaller files."
        )

    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        preview = ", ".join(map(str, missing[:15]))

        if len(missing) > 15:
            preview += f", and {len(missing) - 15} more"

        raise ValueError(
            f"{len(missing)} required model feature(s) are missing: "
            f"{preview}. Upload a dataset compatible with the saved "
            "model's original feature format."
        )

    # Use only the saved model's required features.
    X = df[feature_columns].copy()

    # Detect missing cells before numeric conversion.
    missing_counts = X.isna().sum()
    missing_columns = missing_counts[missing_counts > 0]

    if not missing_columns.empty:
        details = ", ".join(
            f"{col} ({count} missing)"
            for col, count in missing_columns.items()
        )

        raise ValueError(
            "Missing values were found in: "
            + details
            + ". Correct these cells before uploading again."
        )

    # Convert feature values to numbers.
    for col in feature_columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        invalid_mask = converted.isna()

        if invalid_mask.any():
            count = int(invalid_mask.sum())

            raise ValueError(
                f"Column '{col}' contains {count} blank or "
                "non-numeric value(s). Check the affected cells."
            )

        X[col] = converted

    try:
        values = X.to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Some feature values cannot be represented as numbers."
        ) from exc

    if not np.isfinite(values).all():
        raise ValueError(
            "Infinite or invalid numeric values were found. "
            "Correct the affected cells before continuing."
        )

    return X


def predict_transactions(X):
    """Use the original scaler, model and decision threshold."""

    X_model = X.copy()

    # Preserve the original preprocessing behavior.
    if "Time" in feature_columns and "Amount" in feature_columns:
        X_model[["Time", "Amount"]] = scaler.transform(
            X_model[["Time", "Amount"]]
        )

    probabilities_all = model.predict_proba(X_model)

    if (
        probabilities_all.ndim != 2
        or probabilities_all.shape[0] != len(X_model)
        or probabilities_all.shape[1] < 2
    ):
        raise ValueError(
            "The model returned an unexpected prediction format."
        )

    # Preserve the original assumption that column 1 is the
    # fraud-class probability.
    probabilities = probabilities_all[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "The model returned invalid probability values."
        )

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            "The model returned probabilities outside the 0–1 range."
        )

    predictions = (probabilities >= threshold).astype(int)

    return probabilities, predictions

# --------------------------------------------------
# HEADER
# --------------------------------------------------

LOGO_URI = ""

if LOGO_PATH.exists():
    try:
        LOGO_URI = (
            "data:image/svg+xml;base64,"
            + base64.b64encode(
                LOGO_PATH.read_bytes()
            ).decode("utf-8")
        )
    except OSError:
        LOGO_URI = ""

if LOGO_URI:
    brand_html = (
        f'<img src="{LOGO_URI}" alt="FRAUDLENS logo" '
        'style="width:100%;max-width:440px;height:auto;'
        'display:block;margin:12px auto 18px;">'
    )
else:
    brand_html = "<h1>FRAUDLENS</h1>"

st.markdown(f"""
<div class="hero">
    <div class="eyebrow">INTELLIGENCE • ANALYTICS • SECURITY</div>
    {brand_html}
    <p><b>Fraud Risk Intelligence Platform</b></p>
    <p>
        Explore transaction patterns, estimate fraud risk, and turn
        model predictions into clear, actionable insights.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.caption("Fraud Risk Intelligence")
    st.divider()

    st.markdown("## 🔐 FRAUDLENS")
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
# DASHBOARD OVERVIEW
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Dashboard Overview</div>',
    unsafe_allow_html=True,
)

st.caption("Upload compatible transaction data to begin your analysis.")

overview1, overview2, overview3 = st.columns(3)

overview1.metric(
    "Model",
    bundle.get("model_name", "ML model"),
)
overview2.metric("Risk threshold", f"{threshold:.2%}")
overview3.metric("Required features", len(feature_columns))

st.divider()

# --------------------------------------------------
# FIRST-TIME USER GUIDE
# --------------------------------------------------

with st.expander(
    "👋 First time here? Read this quick guide",
    expanded=True,
):
    st.markdown("""
    ### How to use FRAUDLENS

    **Step 1 — Upload or explore**

    Select a compatible transaction CSV, or use the sample dataset
    to explore the dashboard.

    **Step 2 — Check your data**

    FRAUDLENS checks that required features exist and contain
    valid numeric values.

    **Step 3 — Review predictions**

    Examine the risk summary, charts and transaction-level results.

    **Step 4 — Download results**

    Download the analysis as a CSV file for further review.

    ### CSV format requirements

    - Include every feature required by the saved model.
    - The original dataset commonly uses `Time`, `V1`–`V28`
      and `Amount`. The saved model's feature list is authoritative.
    - Required features must contain numeric, finite values.
    - Keep the original feature meanings, units and preprocessing
      assumptions unchanged.
    - Extra columns are retained in the results but are not used
      as model inputs unless required by the saved model.

    **Important:** a high-risk prediction is not proof of fraud,
    and a low-risk prediction is not a guarantee of safety.
    """)

# --------------------------------------------------
# DATA INPUT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📂 Transaction Workspace</div>',
    unsafe_allow_html=True,
)

upload_col, sample_col = st.columns([1.5, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload transaction data",
        type=["csv"],
        help=(
            "Upload a CSV containing every required model feature. "
            "The original dataset commonly uses Time, V1–V28 and Amount."
        ),
    )

with sample_col:
    st.markdown("#### 🧪 Explore the demo")

    st.write(
        "New to the dashboard? Load the sample dataset to explore "
        "the results without uploading your own file."
    )

    sample_clicked = st.button(
        "✨ Try Sample Transactions",
        type="primary",
        use_container_width=True,
    )

# Persist the sample selection across Streamlit reruns.
if "use_sample_transactions" not in st.session_state:
    st.session_state["use_sample_transactions"] = False

if uploaded_file is not None:
    st.session_state["use_sample_transactions"] = False

if sample_clicked:
    st.session_state["use_sample_transactions"] = True

data = None
source_name = None

if uploaded_file is not None:
    source_name = uploaded_file.name

    try:
        data = read_csv_safely(uploaded_file)
        st.info(f"📄 Selected file: {source_name}")

    except ValueError as exc:
        st.error(str(exc))
        st.stop()

elif st.session_state["use_sample_transactions"]:
    source_name = SAMPLE_PATH.name

    if not SAMPLE_PATH.exists():
        st.error(
            "The sample dataset was not found. Place "
            "`sample_transactions.csv` in the same folder as `app.py`."
        )
        st.stop()

    try:
        data = read_csv_safely(SAMPLE_PATH)
        st.success("🌟 Sample dataset loaded successfully.")

    except ValueError as exc:
        st.error(f"Sample dataset problem: {exc}")
        st.stop()

# --------------------------------------------------
# VALIDATION AND PREDICTION
# --------------------------------------------------

if data is not None:
    st.divider()

    st.markdown(
        '<div class="section-title">🔍 Analysis Results</div>',
        unsafe_allow_html=True,
    )

    try:
        X_input = validate_features(data)

    except ValueError as exc:
        st.error(str(exc))
        st.info(
            "Correct the listed problems in your CSV and upload it again."
        )
        st.stop()

    st.success(
        f"✅ Data checks passed — {len(X_input):,} transactions are ready."
    )

    try:
        probabilities, predictions = predict_transactions(X_input)

        results = data.copy()

        results["Fraud Probability (%)"] = (
            probabilities * 100
        ).round(2)

        results["Predicted Fraud"] = predictions

        results["Risk Level"] = np.where(
            predictions == 1,
            "High Risk",
            "Low Risk",
        )

        total = len(results)
        flagged = int(predictions.sum())
        low_risk = total - flagged
        avg_probability = float(probabilities.mean() * 100)

        # ------------------------------------------
        # RISK SUMMARY
        # ------------------------------------------

        st.markdown("#### 📌 Risk Summary")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Transactions analyzed", f"{total:,}")
        m2.metric("⚠️ High-risk flags", f"{flagged:,}")
        m3.metric("✓ Low-risk predictions", f"{low_risk:,}")
        m4.metric(
            "Average estimated probability",
            f"{avg_probability:.2f}%",
        )

        if flagged == 0:
            st.success(
                "No transactions were flagged at the current threshold. "
                "This does not guarantee that every transaction is safe."
            )

        elif flagged == total:
            st.warning(
                "Every transaction was flagged as high risk. Review "
                "the input data and predictions carefully. A flag is "
                "not proof of fraud."
            )

        else:
            st.warning(
                f"The model flagged {flagged:,} of {total:,} "
                "transactions for further review. A prediction is "
                "not a confirmed fraud finding."
            )

        # ------------------------------------------
        # VISUAL ANALYTICS
        # ------------------------------------------

        st.markdown("#### 📈 Visual Insights")

        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown("**Risk distribution**")

            risk_counts = pd.DataFrame(
                {"Transactions": [low_risk, flagged]},
                index=["Low Risk", "High Risk"],
            )

            st.bar_chart(risk_counts)

        with chart2:
            st.markdown("**Estimated probability by transaction**")

            chart_data = pd.DataFrame({
                "Estimated probability (%)": (
                    probabilities[:100] * 100
                )
            })

            chart_data.index = range(1, len(chart_data) + 1)

            st.line_chart(chart_data)

            if total > 100:
                st.caption("Chart displays the first 100 transactions.")

        # ------------------------------------------
        # TRANSACTION RESULTS
        # ------------------------------------------

        st.markdown("#### 🧾 Transaction-Level Results")

        risk_filter = st.selectbox(
            "Filter by risk level",
            ["All transactions", "High Risk", "Low Risk"],
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

        st.caption(
            f"Showing {len(display_results):,} of {total:,} transactions."
        )

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )

        # ------------------------------------------
        # DOWNLOAD RESULTS
        # ------------------------------------------

        st.download_button(
            "⬇️ Download Analysis Results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="fraudlens_fraud_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "Risk labels use the saved model's decision threshold. "
            "Probability estimates and model performance depend on "
            "the training data and should be validated against "
            "verified outcomes."
        )

    except Exception as exc:
        st.error(
            "The analysis could not be completed. Verify that the "
            "dataset matches the saved model's expected format and "
            "that its preprocessing requirements are satisfied."
        )
        st.caption(
            f"Technical detail: {type(exc).__name__}: {exc}"
        )

else:
    st.info(
        "👋 Welcome! Upload a compatible CSV or select "
        "'Try Sample Transactions' to explore FRAUDLENS."
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

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    '<p class="small-muted" style="text-align:center;">'
    'FRAUDLENS • FRAUD RISK INTELLIGENCE<br>'
    'Academic portfolio project | Not for production banking decisions'
    '</p>',
    unsafe_allow_html=True,
)
