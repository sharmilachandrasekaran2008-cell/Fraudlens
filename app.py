import base64
from io import BytesIO
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
# DOWNLOAD REPORT HELPERS
# --------------------------------------------------
def make_excel_report(results, total, flagged, avg_probability, threshold):
    output = BytesIO()

    summary = pd.DataFrame({
        "Metric": [
            "Transactions analyzed",
            "High-risk flags",
            "Low-risk predictions",
            "Average estimated fraud probability (%)",
            "Decision threshold (%)",
        ],
        "Value": [
            total,
            flagged,
            total - flagged,
            round(avg_probability, 2),
            round(threshold * 100, 2),
        ],
    })

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        results.to_excel(writer, sheet_name="Transactions", index=False)

    output.seek(0)
    return output.getvalue()


def make_pdf_report(results, total, flagged, avg_probability, threshold):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )

    output = BytesIO()

    doc = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#0C766E"),
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=8,
    ))

    story = [
        Paragraph("FRAUDLENS", styles["ReportTitle"]),
        Paragraph("Fraud Risk Intelligence — Analysis Report",
                  styles["Heading2"]),
        Spacer(1, 5 * mm),
        Paragraph(f"Transactions analyzed: {total:,}", styles["Normal"]),
        Paragraph(f"High-risk flags: {flagged:,}", styles["Normal"]),
        Paragraph(f"Low-risk predictions: {total - flagged:,}",
                  styles["Normal"]),
        Paragraph(
            f"Average estimated fraud probability: {avg_probability:.2f}%",
            styles["Normal"]
        ),
        Paragraph(
            f"Decision threshold: {threshold:.2%}",
            styles["Normal"]
        ),
        Spacer(1, 5 * mm),
        Paragraph("Transaction Results (first 50 rows)", styles["Heading2"]),
    ]

    # Keep the PDF readable even when the source has many model features.
    preferred_columns = [
        "Fraud Probability (%)",
        "Predicted Fraud",
        "Risk Level",
        "Recommended Action",
    ]
    visible_columns = [
        col for col in preferred_columns if col in results.columns
    ]

    pdf_data = [["Transaction #"] + visible_columns]

    for row_number, (_, row) in enumerate(
        results[visible_columns].head(50).iterrows(), start=1
    ):
        values = [str(row_number)]
        for col in visible_columns:
            value = row[col]
            values.append("" if pd.isna(value) else str(value))
        pdf_data.append(values)

    table = Table(pdf_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102D3A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C8D3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor("#EFF5F7")
        ]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))

    story.extend([
        table,
        Spacer(1, 5 * mm),
        Paragraph(
            "Important: These are machine-learning estimates, not "
            "confirmed fraud findings. This PDF displays at most 50 "
            "transactions; use the CSV or Excel export for the full results.",
            styles["Normal"]
        ),
    ])

    doc.build(story)
    output.seek(0)
    return output.getvalue()

# --------------------------------------------------
# FRAUDLENS | PREMIUM DARK NAVY + TEAL THEME
# --------------------------------------------------
st.markdown("""
<style>
:root {
    --bg: #07111F;
    --panel: #0D1B2B;
    --panel-raised: #112438;
    --border: #20384D;
    --teal: #36E0C0;
    --text: #F2F7FC;
    --muted: #9BB0C5;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #123047 0%, transparent 35%),
        linear-gradient(145deg, #06101D 0%, #091625 52%, #0A1928 100%);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(7, 17, 31, 0.85);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091523 0%, #0D1D2D 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: -0.4px;
}

p, label, .stMarkdown {
    color: inherit;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 34px 36px;
    border: 1px solid #28516A;
    border-radius: 24px;
    background:
        radial-gradient(ellipse at 90% 0%, #15514F 0%, transparent 42%),
        linear-gradient(120deg, #10243A 0%, #102D3A 55%, #0B3436 100%);
    margin-bottom: 28px;
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.18);
}

.hero::after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    right: -100px;
    bottom: -140px;
    border: 1px solid rgba(54, 224, 192, 0.3);
    border-radius: 50%;
    box-shadow:
        0 0 0 25px rgba(54, 224, 192, 0.035),
        0 0 0 50px rgba(54, 224, 192, 0.025);
    pointer-events: none;
}

.eyebrow {
    color: var(--teal);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero h1 {
    color: #F6FAFF;
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;
    margin: 12px 0 8px;
    line-height: 1.12;
}

.hero p {
    color: #BDD0DF;
    font-size: 15px;
    line-height: 1.7;
    max-width: 760px;
}

.hero-subtitle {
    color: #F1F8FC !important;
    font-size: 17px !important;
    font-weight: 700;
    margin-bottom: 5px !important;
}

.hero-badge {
    display: inline-block;
    margin-top: 15px;
    padding: 7px 12px;
    border: 1px solid rgba(54, 224, 192, 0.35);
    border-radius: 999px;
    background: rgba(54, 224, 192, 0.08);
    color: #79F0D7;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

.section-title {
    color: #F3F8FE;
    font-size: 23px;
    font-weight: 750;
    letter-spacing: -0.5px;
    margin-top: 8px;
    margin-bottom: 3px;
}

.section-subtitle {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 18px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #12263A 0%, #0C1A2A 100%);
    border: 1px solid #284258;
    border-radius: 17px;
    padding: 20px 21px;
    min-height: 118px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: #3ABDA9;
    transform: translateY(-2px);
}

div[data-testid="stMetricLabel"] {
    color: #A8BED1;
    font-size: 13px;
}

div[data-testid="stMetricValue"] {
    color: #F4F9FF;
    font-weight: 750;
    font-size: clamp(21px, 2.3vw, 30px);
}

div[data-testid="stMetricDelta"] {
    font-size: 12px;
}

div[data-testid="stFileUploader"] {
    background: #0D1D2D;
    border: 1px dashed #3A6679;
    border-radius: 16px;
    padding: 18px;
}

div[data-testid="stFileUploader"] section {
    border: none;
}

div[data-testid="stFileUploaderDropzone"] {
    background: #112438;
    border: 1px dashed #41677C;
    border-radius: 12px;
}

div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--teal);
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 11px;
    min-height: 45px;
    font-weight: 700;
    border: 1px solid #367C7B;
    transition: all 0.2s ease;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(110deg, #32D9B6, #20B89F);
    color: #061923;
    border: 1px solid #48E8CB;
}

.stButton > button[kind="primary"]:hover {
    background: #70F0D8;
    border-color: #70F0D8;
    color: #061923;
}

.stDownloadButton > button {
    background: #102A3B;
    color: #74EBD5;
    border-color: #337C7B;
}

.stDownloadButton > button:hover {
    background: #173C4B;
    color: #FFFFFF;
    border-color: #50DCC3;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #284258;
    border-radius: 14px;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.stSelectbox > div > div,
.stTextInput > div > div {
    background-color: #102237;
    border-color: #304C62;
    border-radius: 10px;
}

[data-testid="stMarkdownContainer"] hr,
hr {
    border-color: #233B50;
}

.small-muted {
    color: #8EA7BD;
    font-size: 12px;
    letter-spacing: 0.3px;
}

div[data-testid="stCaptionContainer"] {
    color: #8EA7BD;
}

[data-testid="stSidebar"] .stMetric {
    background: #102135;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 24px 20px;
        border-radius: 18px;
    }

    .hero h1 {
        font-size: 30px;
    }
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
# FRAUDLENS | PREMIUM HERO HEADER
# --------------------------------------------------
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">INTELLIGENCE / ANALYTICS / SECURITY</div>
    {
        f'<img src="{LOGO_URI}" alt="FRAUDLENS logo" '
        'style="width:100%;max-width:390px;height:auto;'
        'display:block;margin:18px 0 22px;">'
        if LOGO_URI else
        '<h1>FRAUDLENS</h1>'
    }
    <p class="hero-subtitle">Fraud Risk Intelligence Platform</p>
    <p>
        Turn transaction data into meaningful risk insights.
        Explore model-estimated fraud probabilities, identify
        transactions for review, and export your analysis.
    </p>
    <span class="hero-badge">● MACHINE LEARNING RISK ANALYSIS</span>
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
# TOP OVERVIEW | MODEL SNAPSHOT
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Dashboard Overview</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">'
    'Your model configuration at a glance.'
    '</div>',
    unsafe_allow_html=True
)

overview1, overview2, overview3 = st.columns(3)

with overview1:
    st.metric(
        label="MODEL",
        value=bundle.get("model_name", "ML model")
    )

with overview2:
    st.metric(
        label="DECISION THRESHOLD",
        value=f"{threshold:.2%}"
    )

with overview3:
    st.metric(
        label="REQUIRED FEATURES",
        value=f"{len(feature_columns)}"
    )

st.markdown(
    '<p class="small-muted">'
    'MODEL READY &nbsp; • &nbsp; '
    'FEATURE CONFIGURATION LOADED &nbsp; • &nbsp; '
    'PREDICTIONS ARE ESTIMATES'
    '</p>',
    unsafe_allow_html=True
)

st.divider()



# --------------------------------------------------
# DATA INPUT
# --------------------------------------------------

upload_col, sample_col = st.columns([1.5, 1])

st.markdown(
    '<div class="section-title">Transaction Workspace</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">'
    'Import your transaction dataset or explore the dashboard '
    'with the provided demo data.'
    '</div>',
    unsafe_allow_html=True
)

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
 
if "use_sample" not in st.session_state:
    st.session_state["use_sample"] = False

sample_clicked = st.button(
    "✨ Try Sample Transactions",
    type="primary",
    use_container_width=True
)

if sample_clicked:
    st.session_state["use_sample"] = True
  
data = None

if uploaded_file is not None:
    st.session_state["use_sample"] = False
    try:
        data = pd.read_csv(uploaded_file)
        st.info("📄 Your uploaded dataset is selected.")
    except Exception:
        st.error(
            "We couldn't read this file. Please upload a valid CSV file."
        )
        st.stop()
        
elif st.session_state["use_sample"]:
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
        
        results["Recommended Action"] = np.where(
            predictions == 1,
            "Review transaction",
            "Routine monitoring"
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

        
# Results table and risk classification guide
st.markdown("#### 🧾 Transaction-Level Results")

st.markdown("""
<div style="
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
    gap:12px;
    margin:12px 0 18px;
">
    <div style="
        background:#102B2B;
        border:1px solid #287E70;
        border-radius:12px;
        padding:16px;
    ">
        <div style="color:#65E8C9;font-weight:700;">
            HIGH RISK
        </div>
        <div style="color:#D7E6EF;font-size:13px;margin-top:7px;">
            Flagged by the model. Review the transaction using
            verified records before taking action.
        </div>
    </div>
    <div style="
        background:#102438;
        border:1px solid #315C78;
        border-radius:12px;
        padding:16px;
    ">
        <div style="color:#8CCBFF;font-weight:700;">
            LOW RISK
        </div>
        <div style="color:#D7E6EF;font-size:13px;margin-top:7px;">
            Not flagged at the current threshold. This does not
            rule out fraud; continue normal monitoring.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

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
            file_name="fraudlens_fraud_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "Risk labels are based on the saved model's decision "
            "threshold. Model performance depends on the training data "
            "and should be validated against verified labels."
        )

    
    except Exception as e:
        st.error(
            "The analysis could not be completed. "
            "Please verify the dataset and model configuration."
        )
        with st.expander("Technical details"):
            st.code(f"{type(e).__name__}: {e}")
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
