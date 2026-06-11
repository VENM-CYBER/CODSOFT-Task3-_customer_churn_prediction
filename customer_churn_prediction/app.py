import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="AI Customer Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# =====================================
# PREMIUM THEME SWITCH IN CORNER
# =====================================

st.markdown("""
    <style>
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(255,255,255,0.1);
        padding: 8px 18px;
        border-radius: 25px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        z-index: 999;
    }
    </style>
""", unsafe_allow_html=True)

theme = st.radio(
    "Theme",
    ["Neo Dark", "Aurora Light"],
    horizontal=True,
    label_visibility="collapsed",
    key="theme_toggle"
)

st.markdown("<div class='theme-toggle'></div>", unsafe_allow_html=True)

# Apply theme colors
if theme == "Neo Dark":
    bg = "#0d1117"
    card = "rgba(22,27,34,0.85)"
    text = "#e6edf3"
    accent = "#00d4ff"
    gradient = "linear-gradient(90deg,#0d1117,#161b22,#1f6feb)"
else:
    bg = "#f0f4f8"
    card = "rgba(255,255,255,0.85)"
    text = "#1a202c"
    accent = "#ff6b6b"
    gradient = "linear-gradient(90deg,#f0f4f8,#e2e8f0,#ff6b6b)"

# =====================================
# PREMIUM UI STYLING
# =====================================

st.markdown(f"""
<style>
.stApp {{
    background:{bg};
    color:{text};
    font-family: 'Segoe UI', sans-serif;
}}
.card {{
    background:{card};
    padding:25px;
    border-radius:20px;
    backdrop-filter: blur(12px);
    box-shadow:0 8px 40px rgba(0,0,0,.25);
}}
.title {{
    text-align:center;
    font-size:56px;
    font-weight:900;
    background: {gradient};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.metric-container {{
    display:flex;
    justify-content:center;
    gap:40px;
    margin-top:20px;
}}
</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================

MODEL_PATH = os.path.join("models", "customer_churn_model.pkl")
model = joblib.load(MODEL_PATH)

# =====================================
# HEADER
# =====================================

st.markdown("<div class='title'>🏦 AI Customer Churn Predictor</div>", unsafe_allow_html=True)
st.write("")

# =====================================
# INPUTS
# =====================================

st.markdown("### 📝 Customer Information")
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 900, 650)
    age = st.slider("Age", 18, 100, 35)
    balance = st.number_input("Balance", value=50000.0)
    salary = st.number_input("Estimated Salary", value=50000.0)

with col2:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure", 0, 10, 5)
    products = st.slider("Products", 1, 4, 2)
    active_member = st.selectbox("Active Member", [1, 0])
    has_card = st.selectbox("Has Credit Card", [1, 0])

# =====================================
# DATAFRAME
# =====================================

sample = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [products],
    "HasCrCard": [has_card],
    "IsActiveMember": [active_member],
    "EstimatedSalary": [salary]
})

# =====================================
# PREDICT SINGLE CUSTOMER
# =====================================

if st.button("🚀 Predict Churn", use_container_width=True):

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Churn Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": accent}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Risk Level
    if probability < 0.3:
        risk = "🟢 LOW RISK"
    elif probability < 0.6:
        risk = "🟠 MEDIUM RISK"
    else:
        risk = "🔴 HIGH RISK"

    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.metric("Risk Level", risk)
    st.markdown("</div>", unsafe_allow_html=True)

    if prediction == 1:
        st.error("⚠ Customer likely to leave")
    else:
        st.success("✅ Customer likely to stay")

    # Customer Profile
    st.markdown("### 📊 Customer Profile")
    profile_col1, profile_col2 = st.columns(2)
    with profile_col1:
        st.write(f"**Age:** {age}")
        st.write(f"**Credit Score:** {credit_score}")
        st.write(f"**Balance:** {balance}")
    with profile_col2:
        st.write(f"**Geography:** {geography}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Products:** {products}")

    # Feature Importance (Fallback)
    st.markdown("### 🔍 Feature Importance")
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": sample.columns,
            "Importance": importance
        }).sort_values("Importance", ascending=True)

        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     title="Model Feature Importance", color="Importance",
                     color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Explainability not available for this model type.")

    # Report Download
    report = pd.DataFrame({
        "Prediction": [prediction],
        "Probability": [probability],
        "Risk": [risk],
        "Time": [datetime.now()]
    })

    st.download_button(
        "📥 Download Report",
        report.to_csv(index=False),
        "prediction_report.csv",
        "text/csv"
    )

    # Session Logging
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append(report)

    st.markdown("### 🗂 Prediction History")
    st.dataframe(pd.concat(st.session_state["history"], ignore_index=True))

# =====================================
# BATCH PREDICTION FEATURE
# =====================================

st.markdown("## 📥 Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload customer dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    preds = model.predict(data)
    probs = model.predict_proba(data)[:, 1]

    data["Prediction"] = preds
    data["Probability"] = probs
    data["Risk"] = pd.cut(probs,
                          bins=[0, 0.3, 0.6, 1],
                          labels=["LOW", "MEDIUM", "HIGH"])

    st.markdown("### 📊 Portfolio Dashboard")

    # Risk Distribution Pie Chart
    fig_pie = px.pie(data, names="Risk", title="Churn Risk Distribution",
                     color="Risk", color_discrete_map={
                         "LOW": "green",
                         "MEDIUM": "orange",
                         "HIGH": "red"
                     })
    st.plotly_chart(fig_pie, use_container_width=True)

    # Probability Histogram
    fig_hist = px.histogram(data, x="Probability", nbins=20,
                            title="Churn Probability Histogram",
                            color="Risk", color_discrete_map={
                                "LOW": "green",
                                "MEDIUM": "orange",
                                "HIGH": "red"
                            })
    st.plotly_chart(fig_hist, use_container_width=True)

    # Download batch results
    st.download_button(
        "📥 Download Batch Results",
        data.to_csv(index=False),
        "batch_predictions.csv",
    )