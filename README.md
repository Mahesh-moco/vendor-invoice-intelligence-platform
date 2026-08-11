# 📦 Vendor Invoice Intelligence Platform

An end-to-end Machine Learning platform that predicts **freight costs** and flags **high-risk vendor invoices** requiring manual review — built with Python, Scikit-learn, and Streamlit.

---

## 🚀 Overview

Manually reviewing every vendor invoice for pricing errors, fraud, or anomalies is slow and error-prone. This project automates two key finance operations tasks using machine learning:

- **Freight Cost Prediction** — estimates expected freight cost from invoice data, helping teams catch overcharges early.
- **Invoice Risk Flagging** — classifies invoices as "safe for auto-approval" or "requires manual approval" based on historical patterns, reducing manual audit workload.

Both models are served through an interactive **Streamlit** web app for real-time predictions.

---

## ✨ Features

- 🚛 **Freight Cost Prediction** — regression model estimating freight cost from invoice dollar amount
- 🚨 **Invoice Risk Flagging** — classification model detecting invoices needing manual review
- ⚡ **Interactive Web Dashboard** — built with Streamlit for real-time, no-code predictions
- 🧠 **Modular Inference Pipeline** — clean separation between model logic and UI for easy updates
- 📊 **Live Results Display** — instant prediction output with metrics and data tables

---

## 🛠️ Tech Stack

| Category         | Tools                          |
|-------------------|--------------------------------|
| Language          | Python                         |
| ML / Data         | Scikit-learn, Pandas, NumPy    |
| Web App           | Streamlit                      |
| Data Storage      | SQLite                         |
| Model Persistence | Pickle (.pkl)                  |

---

## 📁 Project Structure

```
ML_project_2.0/
│
├── app.py                          # Streamlit web app (main entry point)
├── inference/                      # Inference layer — loads models & runs predictions
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── freight_cost_prediction/        # Freight cost model pipeline
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── model_evaluation.py
│   └── models/
│       └── predict_freight_model.pkl
│
├── invoice_flagging/               # Invoice risk flagging model pipeline
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── modeling_evaluation.py
│   └── models/
│       ├── predict_flag_invoice.pkl
│       └── scaler.pkl
│
├── notbooks/                       # Jupyter notebooks (EDA & experimentation)
├── data/                           # Raw/processed data (excluded from repo)
└── README.md
```

---

## ⚙️ How It Works

1. **Freight Cost Prediction**
   User enters an invoice dollar amount → the trained regression model predicts the expected freight cost.

2. **Invoice Risk Flagging**
   User enters invoice details (quantity, dollars, freight, receiving delay, etc.) → the trained classification model predicts whether the invoice is safe for auto-approval or needs manual review.

---

## 🖥️ Running the App Locally

**1. Clone the repository**
```bash
git clone https://github.com/Mahesh-moco/vendor-invoice-intelligence-platform.git
cd vendor-invoice-intelligence-platform
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

**4. Open in browser**
Streamlit will automatically open the app at:
```
http://localhost:8501
```

---

## 📌 Future Improvements

- Add model retraining pipeline with new invoice data
- Add authentication for internal finance team access
- Deploy on Streamlit Cloud / AWS / Azure
- Add SHAP-based explainability for flagged invoices

---

## 👤 Author

**Mahesh**
Built as a machine learning project applying predictive analytics to real-world finance operations.
