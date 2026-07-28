# 💳 Fortress: Credit Card Fraud Detection System

An end-to-end Machine Learning application for detecting fraudulent credit card transactions using **XGBoost**, **FastAPI**, **Streamlit**, and **SHAP Explainable AI**.

The system predicts whether a transaction is **Genuine** or **Fraudulent**, provides the fraud probability, confidence score, risk level, and explains the prediction using SHAP feature importance.

---

## 🚀 Features

- End-to-End Machine Learning Pipeline
- Data Preprocessing and Feature Scaling
- XGBoost Classifier
- FastAPI REST API
- Interactive Streamlit Dashboard
- Fraud Probability Prediction
- Confidence Score
- Risk Level Classification
- SHAP Explainability
- Prediction History
- Transaction Summary
- Modular Project Structure
- Deployment Ready

---

## 📂 Project Structure

```text
credit-card-fraud-detection/

│── data/
│   └── creditcard.csv
│
│── models/
│   ├── xgboost.pkl
│   └── scaler.pkl
│
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
│
│── src/
│   ├── api/
│   │   ├── app.py
│   │   └── predict.py
│   │
│   └── dashboard/
│       ├── app.py
│       ├── api.py
│       └── shap_utils.py
│
│── requirements.txt
│── README.md
```

---

## 📊 Dataset

This project uses the **Credit Card Fraud Detection Dataset**, containing anonymized European credit card transactions.

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Transactions | 284,807 |
| Fraud Cases | 492 |
| Features | 30 |
| Fraud Ratio | 0.172% |

### Features

- Time
- Amount
- V1 – V28 (PCA-transformed features)
- Class (Target)

Target Variable

- **0 → Genuine Transaction**
- **1 → Fraudulent Transaction**

---

## 🧠 Machine Learning Pipeline

1. Load and explore the dataset
2. Scale `Time` and `Amount`
3. Split training and testing data
4. Train an XGBoost classifier
5. Evaluate model performance
6. Save trained model and scaler
7. Serve predictions using FastAPI
8. Build an interactive Streamlit dashboard
9. Explain predictions using SHAP

---

## 🤖 Model

**Algorithm Used**

- XGBoost Classifier

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## 🌐 FastAPI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status |
| `/health` | GET | Health check |
| `/predict` | POST | Predict transaction fraud |

### Sample Request

```json
{
  "Time": 0,
  "Amount": 149.62,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536346,
  "...": "...",
  "V28": -0.021053
}
```

### Sample Response

```json
{
  "prediction": 0,
  "fraud_probability": 0.00023
}
```

---

## 💻 Streamlit Dashboard

The dashboard provides:

- Single Transaction Prediction
- Fraud Probability
- Confidence Score
- Risk Level
- Transaction Summary
- SHAP Feature Importance
- SHAP Waterfall Plot
- Prediction History

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git

cd credit-card-fraud-detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

Run the notebooks in order:

1. Data Preprocessing
2. Model Training
3. Model Evaluation

The trained artifacts will be saved in the `models/` directory.

---

## ▶️ Run FastAPI

```bash
cd src/api

uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## ▶️ Run Streamlit

```bash
cd src/dashboard

streamlit run app.py
```

---

## 📈 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Streamlit
- SHAP
- Matplotlib
- Joblib

---

## 📌 Future Improvements

- Batch prediction using CSV upload
- Real-time transaction monitoring
- User authentication
- Database integration
- Docker support
- Cloud deployment
- Model monitoring
- CI/CD pipeline
- Fraud analytics dashboard
- Automatic model retraining

---

## 👩‍💻 Author

**Richa Pandey**

B.Tech Computer Science & Engineering (AI)

Machine Learning • Data Science • FastAPI • Streamlit • XGBoost

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- Credit Card Fraud Detection Dataset
- Scikit-learn
- XGBoost
- FastAPI
- Streamlit
- SHAP
- Pandas
- NumPy

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub.