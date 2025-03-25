import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os

# --- Page Setup ---
st.set_page_config(page_title="FairDeploy: AI Model Deployment & Monitoring Dashboard", layout="wide")

# Load Dark Mode CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar Uploads ---
st.sidebar.title("🛠️ Upload Your Model & Data (Optional)")

# Upload multiple models
uploaded_models = st.sidebar.file_uploader(
    "Upload Multiple Models (.pkl)", type=["pkl"], accept_multiple_files=True
)

# Upload dataset (single)
uploaded_data = st.sidebar.file_uploader("Upload Dataset (.csv)", type=["csv"])

# Save uploaded models to /models_uploaded/
model_paths = []
if uploaded_models:
    os.makedirs("models_uploaded", exist_ok=True)
    for file in uploaded_models:
        path = f"models_uploaded/{file.name}"
        with open(path, "wb") as f:
            f.write(file.read())
        model_paths.append(path)

# --- Load Model (with Safe Patch) ---
def patch_monotonic(model_obj):
    if not hasattr(model_obj, 'monotonic_cst'):
        setattr(model_obj, 'monotonic_cst', None)
    return model_obj

try:
    if model_paths:
        selected_model_path = st.sidebar.selectbox("Select Model Version", model_paths)
        model = joblib.load(selected_model_path)
        model = patch_monotonic(model)
        st.sidebar.success(f"✅ Loaded: {os.path.basename(selected_model_path)}")
    else:
        model = joblib.load('models/partner_model.pkl')
        model = patch_monotonic(model)
        st.sidebar.info("Using sample model.")
except Exception as e:
    st.sidebar.error(f"❌ Model load error: {e}")
    st.stop()

# --- Load Data ---
try:
    if uploaded_data is not None:
        df = pd.read_csv(uploaded_data)
        st.sidebar.success("✅ Custom dataset loaded.")
    else:
        df = pd.read_csv('data/sample_dataset.csv')
        st.sidebar.info("Using sample dataset.")
except Exception as e:
    st.sidebar.error(f"❌ Data load error: {e}")
    st.stop()

# --- Prepare Features ---
required_columns = {'label', 'gender'}
if not required_columns.issubset(df.columns):
    st.error(f"⚠️ Data Error: Dataset must include columns: {required_columns}")
    st.stop()

X = df.drop(columns=['label', 'gender'])
y_true = df['label']

# --- Predict Safely ---
try:
    if not hasattr(model, 'monotonic_cst'):
        model.monotonic_cst = None

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"⚠️ Prediction Error: {e}")
    st.stop()

# --- Metrics Calculation ---
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc = roc_auc_score(y_true, y_prob)

# --- Header ---
st.markdown("""
    <div class='title'>
        🚀 Executive AI Insights Dashboard
    </div>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align:center;'>Performance, Fairness, Impact — Actionable Insights for Leadership</p>", unsafe_allow_html=True)

# --- Fairness Setup ---
spd_values = [0.08, 0.09, 0.12, 0.15]
weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
latest_spd = spd_values[-1]
monthly_patients = 100
cost_per_patient = 500
bias_impact = int(latest_spd * monthly_patients)
bias_cost = bias_impact * cost_per_patient
recommendation = "Retrain model with balanced gender data in 2 weeks." if latest_spd > 0.1 else "No action needed. Continue monthly fairness audits."

# --- Tabs ---
tab_overview, tab_fairness, tab_forecast, tab_explain, tab_summary = st.tabs(["📊 Overview", "⚖️ Fairness", "🔮 Forecast", "🧠 Explainability", "🗾 Summary"])

# --- Overview Tab ---
with tab_overview:
    st.markdown("<h2 style='text-align:center;'>Key Performance Indicators (KPIs)</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="large")
    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("F1 Score", f"{f1:.2f}")
    col3.metric("ROC-AUC", f"{roc:.2f}")

    past_accuracy = 0.995
    accuracy_change = acc - past_accuracy

    if accuracy_change < -0.01:
        st.error(f"⚠️ Accuracy dropped by {abs(accuracy_change)*100:.1f}%. Retrain recommended.")
    elif accuracy_change < 0:
        st.warning(f"⚠️ Accuracy decreased by {abs(accuracy_change)*100:.1f}%. Monitor closely.")
    else:
        st.success(f"✅ Accuracy stable or improved (+{accuracy_change*100:.1f}%).")

    st.markdown(f"<div class='insight-box'>📊 Action: {recommendation}</div>", unsafe_allow_html=True)

# --- Fairness Tab ---
with tab_fairness:
    st.markdown("<h3>📉 Fairness Trend</h3>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=spd_values, mode='lines+markers', line=dict(color='orange')))
    fig.add_hline(y=0.1, line_dash="dash", line_color="red", annotation_text="Bias Threshold")

    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if latest_spd <= 0.1:
        st.success(f"✅ Risk Level: LOW — SPD = {latest_spd:.2f} within threshold.")
    elif latest_spd <= 0.15:
        st.warning(f"⚠️ Risk Level: MEDIUM — SPD = {latest_spd:.2f} approaching threshold.")
    else:
        st.error(f"❌ Risk Level: HIGH — SPD = {latest_spd:.2f} exceeds threshold. Action required.")

    st.markdown(f"<div class='insight-box'>📉 Bias Context: {latest_spd:.2f} SPD → ~{bias_impact} biased outcomes/month → Estimated Cost: ${bias_cost:,}/month.</div>", unsafe_allow_html=True)

# --- Forecast Tab ---
with tab_forecast:
    st.subheader("Impact Scenario Forecast")
    forecast_spd = st.slider("Forecast: If Bias (SPD) increases to...", 0.10, 0.30, 0.20, 0.01)
    forecast_impact = int(forecast_spd * monthly_patients)
    forecast_cost = forecast_impact * cost_per_patient

    colF1, colF2 = st.columns(2)
    colF1.metric("Estimated Biased Outcomes", f"{forecast_impact} patients/month")
    colF2.metric("Estimated Cost", f"${forecast_cost:,}/month")

    st.warning(f"⚠️ If SPD reaches {forecast_spd:.2f}, expected cost: ${forecast_cost:,}/month.")

# --- Explainability Tab ---
with tab_explain:
    st.subheader("SHAP Explainability")
    explainer = shap.TreeExplainer(model)

    index = st.number_input("Select Patient Case (1-568)", min_value=0, max_value=len(X)-1, value=0)
    sample_input = X.iloc[[index]]
    st.dataframe(sample_input)

    sample_pred = model.predict(sample_input)[0]
    sample_prob = model.predict_proba(sample_input)[0][1]
    st.success(f"Prediction: {'Malignant' if sample_pred == 1 else 'Benign'} (Confidence: {sample_prob:.2f})")

    shap_values = explainer.shap_values(sample_input)
    top_features = pd.Series(shap_values[1][0], index=sample_input.columns).abs().sort_values(ascending=False).head(3).index.tolist()
    reasoning = ", ".join(top_features)
    st.markdown(f"<div class='insight-box'>📝 Reason: Prediction due to {reasoning}.</div>", unsafe_allow_html=True)

# --- Summary Tab ---
with tab_summary:
    st.markdown("<h2 style='text-align:center;'>🗾 Executive Leadership Summary</h2>", unsafe_allow_html=True)
    st.markdown(f"""
        <ul style='line-height: 1.6; font-size: 16px;'>
            <li><strong>Accuracy:</strong> {acc:.2f}</li>
            <li><strong>F1 Score:</strong> {f1:.2f}, <strong>ROC-AUC:</strong> {roc:.2f}</li>
            <li><strong>Fairness:</strong> SPD = {latest_spd:.2f} (Threshold: 0.10)</li>
            <li><strong>Bias Cost:</strong> ${bias_cost:,}/month</li>
            <li><strong>Forecast SPD {forecast_spd:.2f}:</strong> Est. Cost = ${forecast_cost:,}/month</li>
            <li><strong>Recommendation:</strong> {recommendation}</li>
        </ul>
    """, unsafe_allow_html=True)

    summary_text = f"""
This week:
- Accuracy: {acc:.2f}
- F1 Score: {f1:.2f}, ROC-AUC: {roc:.2f}
- Fairness: SPD {latest_spd:.2f} (Threshold 0.10)
- Bias Cost: ${bias_cost:,}/month
- Forecast (SPD {forecast_spd:.2f}): ${forecast_cost:,}/month
- Recommendation: {recommendation}
"""
    st.download_button("📅 Download Full Summary", summary_text, file_name="AI_Leadership_Summary.txt")

# --- Footer ---
st.markdown("<p class='footer'>AI Insights Dashboard • Empowering Leadership Decisions</p>", unsafe_allow_html=True)
