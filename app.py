import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
import os

# --- Page Setup ---
st.set_page_config(page_title=" FairDeploy: AI Model Deployment & Monitoring Dashboard", layout="wide")

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

# Model Version Selection
# --- Load Model (with Patch Fix) ---
try:
    if model_paths:
        selected_model_path = st.sidebar.selectbox("Select Model Version", model_paths)
        model = joblib.load(selected_model_path)
        st.sidebar.success(f"✅ Loaded: {os.path.basename(selected_model_path)}")
    else:
        model = joblib.load('models/partner_model.pkl')
        st.sidebar.info("Using sample model.")

    # 🛠️ Patch: Add dummy monotonic_cst attribute if missing
    if not hasattr(model, 'monotonic_cst'):
        model.monotonic_cst = None  # Or []
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

# --- : Export to ONNX (Edge AI) ---
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

if st.sidebar.button("⚡ Export Model to ONNX"):
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open("models/partner_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    st.sidebar.success("✅ Model exported to models/partner_model.onnx")


# --- Tabs ---
tab_overview, tab_fairness, tab_forecast, tab_explain, tab_summary = st.tabs(["📊 Overview", "⚖️ Fairness", "🔮 Forecast", "🧠 Explainability", "🗾 Summary"])

# --- Overview Tab ---
with tab_overview:
    st.markdown("<h2 style='text-align:center;'>Key Performance Indicators (KPIs)</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="large")

    col1.markdown(f"""
    <div class='kpi-card'>
        <h4>Accuracy</h4>
        <p>{acc:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class='kpi-card'>
        <h4>F1 Score</h4>
        <p>{f1:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class='kpi-card'>
        <h4>ROC-AUC</h4>
        <p>{roc:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Model Drift Detection</h3>", unsafe_allow_html=True)
    past_accuracy = 0.995
    accuracy_change = acc - past_accuracy

    if accuracy_change < -0.01:
        st.markdown(f"<div class='risk-high'>⚠️ Accuracy dropped by {abs(accuracy_change)*100:.1f}%. Retrain recommended.</div>", unsafe_allow_html=True)
    elif accuracy_change < 0:
        st.markdown(f"<div class='risk-medium'>⚠️ Accuracy decreased by {abs(accuracy_change)*100:.1f}%. Monitor closely.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='risk-low'>✅ Accuracy stable or improved (+{accuracy_change*100:.1f}%).</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='insight-box'>
        📊 Action: {recommendation}
    </div>
    """, unsafe_allow_html=True)

# --- Fairness Tab ---
with tab_fairness:
    st.markdown("<h3>📉 Fairness Trend</h3>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=spd_values, mode='lines+markers', line=dict(color='orange')))
    fig.add_hline(y=0.1, line_dash="dash", line_color="red", annotation_text="Bias Threshold")

    fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#1f2937",
    plot_bgcolor="#1f2937",
    font=dict(color="#f0f0f0"),
    height=300,
    margin=dict(l=20, r=20, t=30, b=20),
    yaxis=dict(
        range=[0, 0.2],
        title=dict(text="SPD (Bias)", font=dict(size=12)),  # Correct format
        tickfont=dict(size=10)
    ),
    xaxis=dict(
        title=dict(text="Week", font=dict(size=12)),  # Optional: style X title too
        tickfont=dict(size=10)
    ),
    showlegend=False
)


    # Wrap chart in styled container (optional)
    st.markdown("<div style='background:#1f2937;padding:10px;border-radius:10px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


    if latest_spd <= 0.1:
        st.markdown(f"<div class='risk-low'>✅ Risk Level: LOW — SPD = {latest_spd:.2f} within threshold.</div>", unsafe_allow_html=True)
    elif latest_spd <= 0.15:
        st.markdown(f"<div class='risk-medium'>⚠️ Risk Level: MEDIUM — SPD = {latest_spd:.2f} approaching threshold.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='risk-high'>❌ Risk Level: HIGH — SPD = {latest_spd:.2f} exceeds threshold. Action required.</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='insight-box'>📉 Bias Context: {latest_spd:.2f} SPD → ~{bias_impact} biased outcomes/month → Estimated Cost: ${bias_cost:,}/month.</div>", unsafe_allow_html=True)

    st.subheader("Bias Mitigation Simulator (Reweighing)")
    if st.button("Apply Bias Mitigation"):
        data_bld = BinaryLabelDataset(df=df, label_names=['label'], protected_attribute_names=['gender'])
        reweigher = Reweighing(unprivileged_groups=[{'gender': 0}], privileged_groups=[{'gender': 1}])
        data_transf = reweigher.fit_transform(data_bld)
        df['weights'] = data_transf.instance_weights
        mitigated_spd = abs(df[df['gender'] == 0]['label'].mean() - df[df['gender'] == 1]['label'].mean())
        st.error(f"Before: Bias (SPD) = {latest_spd:.2f}")
        st.success(f"After Reweighing: Bias (SPD) = {mitigated_spd:.2f}")
    else:
        st.info("Apply mitigation to observe reduction in bias.")

# --- Forecast Tab ---
with tab_forecast:
    st.subheader("Impact Scenario Forecast")
    st.markdown("Simulate potential bias impact based on future SPD values. Useful for risk planning.")

    forecast_spd = st.slider("Forecast: If Bias (SPD) increases to...", 0.10, 0.30, 0.20, 0.01)
    forecast_impact = int(forecast_spd * monthly_patients)
    forecast_cost = forecast_impact * cost_per_patient

    colF1, colF2 = st.columns(2)
    colF1.metric("Estimated Biased Outcomes", f"{forecast_impact} patients/month")
    colF2.metric("Estimated Cost", f"${forecast_cost:,}/month")

    st.warning(f"⚠️ If SPD reaches {forecast_spd:.2f}, expected cost: ${forecast_cost:,}/month.")

    st.divider()
    st.info("📌 Insight: Forecast helps prioritize fairness mitigation budget planning and intervention timing.")

# --- Explainability Tab ---
with tab_explain:
    st.subheader("SHAP Explainability")
    explainer = shap.TreeExplainer(model)
    st.markdown("Understand *why* the model made a decision for a specific case.")

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

    shap_html = shap.force_plot(explainer.expected_value[1], shap_values[1], sample_input, matplotlib=False)
    shap.save_html("reports/temp_force_plot.html", shap_html)
    with open("reports/temp_force_plot.html", "r") as f:
        force_html = f.read()
    with st.expander("View Interactive SHAP Force Plot"):
        st.components.v1.html(force_html, height=350)

    st.divider()
    st.info("🔍 Insight: SHAP ensures transparency and builds trust in AI decisions. Useful for audits.")

# --- Summary Tab ---
with tab_summary:
    st.markdown("<h2 style='text-align:center;'>🗾 Executive Leadership Summary</h2>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='insight-box' style='margin-top:20px;'>
        <ul style='padding-left: 20px; line-height: 1.6; font-size: 16px;'>
            <li><strong>Accuracy:</strong> {acc:.2f}</li>
            <li><strong>F1 Score:</strong> {f1:.2f}, <strong>ROC-AUC:</strong> {roc:.2f}</li>
            <li><strong>Fairness:</strong> SPD = {latest_spd:.2f} <em>(Threshold: 0.10)</em></li>
            <li><strong>Bias Cost:</strong> ${bias_cost:,}/month</li>
            <li><strong>Forecast SPD {forecast_spd:.2f}:</strong> Est. Cost = ${forecast_cost:,}/month</li>
            <li><strong>Recommendation:</strong> {recommendation}</li>
        </ul>
    </div>
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
