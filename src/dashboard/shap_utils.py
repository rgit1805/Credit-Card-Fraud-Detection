import os
import joblib
import shap
import pandas as pd

# Current file: src/dashboard/shap_utils.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up TWO levels: src/dashboard -> src -> project root
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))

explainer = shap.TreeExplainer(model)

def explain_prediction(transaction):
    df = pd.DataFrame([transaction])
    shap_values = explainer(df)
    return shap_values