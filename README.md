# 🚀 FairDeploy: AI Model Deployment & Monitoring Dashboard

**FairDeploy** is a professional AI observability dashboard designed to simulate a real-world AI model deployment environment. It offers performance insights, fairness analysis, explainability, cost impact simulation, and CI/CD integration — tailored for leadership and enterprise stakeholders.

---

# 🚀 Live Demo
👉 [Launch FairDeploy on Streamlit Cloud](https://fairdeploy.streamlit.app)

---

## 📊 Key Features

| Feature                 | Description                                                                       |
|-------------------------|------------------------------------------------------------------------------------|
| **Model & Data Upload** | Upload custom AI models (.pkl) and datasets (.csv) or use sample assets           |
| **Performance Metrics** | Real-time Accuracy, F1 Score, ROC-AUC with drift alerts                           |
| **Fairness Analysis**   | Weekly bias trend (SPD), cost impact calculator, mitigation simulator             |
| **Forecasting**         | Predict future bias impact and costs for risk planning                            |
| **Explainability**      | SHAP-based case-level interpretability with interactive force plots (via scripts) |
| **Executive Summary**   | Downloadable report summarizing KPIs, bias impact, and recommendations            |                      
| **AI Observability**    | Logs every prediction with audit logs viewer (via scripts)                        |
| **ONNX Export**         | Export models for edge/cloud AI with one click (via scripts)                      |
| **CI/CD Deployment**    | Auto-deploy pipeline using GitHub Actions and Docker (depoly.yml)                 |
| **Cloud Upload (Optional)** | Push model and logs to cloud storage (simulated)                              |

---

## ❓ Why FairDeploy?

✅ Most AI dashboards stop at metrics.  
✅ FairDeploy goes further: bias, transparency, cost impact, and cloud-readiness.  
✅ Designed for real-world leadership use cases, not just experimentation.

---

## 🧑‍💻 Project Structure

```
FairDeploy_AI_Model_Deployment/
│
├── app.py                          # Streamlit Dashboard App
├── requirements.txt                # Python dependencies
├── dockerfile                      # Docker container setup
│
├── models/                         # AI model files (.pkl, .onnx)
├── data/                           # Sample dataset (.csv)
├── deployments/                    # Metadata + logs for model deployment
├── logs/                           # Prediction audit logs
├── reports/                        # SHAP explainability + fairness reports
├── style.css                       # Dark mode + styled UI
│
├── notebooks/                      # Jupyter notebooks (simulation steps)
│   ├── 1_model_upload_simulation.ipynb
│   ├── 2_fairness_analysis_aif360.ipynb
│   ├── 3_shap_explainability.ipynb
│   └── 4_deployment_monitoring_simulation.ipynb
│
└── .github/workflows/deploy.yml    # CI/CD GitHub Actions pipeline
```

---

## 🏃‍♀️ Run Locally

### 1. Clone Repo

```bash
git clone https://github.com/Akshita7844/FairDeploy-AI-Model-Deployment-Monitoring-Dashboard.git
cd FairDeploy-AI-Model-Deployment-Monitoring-Dashboard
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t fairdeploy .
```

### Run Container

```bash
docker run -p 8501:8501 fairdeploy
```

Dashboard available at [http://localhost:8501](http://localhost:8501)

---

## 🛠️ CI/CD Automation

### GitHub Actions Pipeline

- Auto-installs dependencies
- Runs Streamlit app in headless mode
- Confirms successful startup + logs
- Auto-push Docker image to DockerHub

YAML: `.github/workflows/deploy.yml`

---

## 🌐 DockerHub (Public Image)

Pull Docker image:

```bash
docker pull akshita7844/fairdeploy:latest
```

---

## 🔍 Notebooks Showcase

| Notebook                                   | Description                                  |
|-------------------------------------------|----------------------------------------------|
| `1_model_upload_simulation.ipynb`         | Simulates model/data upload process          |
| `2_fairness_analysis_aif360.ipynb`        | Bias analysis + reweighing with AIF360       |
| `3_shap_explainability.ipynb`             | SHAP explainability for AI decisions         |
| `4_deployment_monitoring_simulation.ipynb`| Logs, monitoring, and ONNX export simulation |

---

## 💡 Skills Demonstrated

- AI Observability (audit logs, drift detection)
- Fairness in AI (bias metrics, mitigation)
- Explainability (SHAP, transparency)
- Edge AI (ONNX export, deployment simulation)
- CI/CD (GitHub Actions, Docker, Automation)
- Cloud Readiness (DockerHub, Cloud Push)
- Business Impact Forecasting (Cost, Risk Analysis)

---

## 📢 About the Author

Built by **Akshita Mishra** 
Focus: AI Enablement, Deployment, Partner Innovation.

GitHub: [Akshita7844](https://github.com/Akshita7844)  


---

## ⭐ License

MIT License. Free to use, share, and adapt with credit.
