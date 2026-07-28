import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Add src to python path to import configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import config

def train_and_evaluate_models():
    """
    Loads preprocessed datasets, balances classes using SMOTE, trains Logistic Regression,
    Random Forest, XGBoost, and LightGBM, evaluates them on multiple metrics,
    and saves the best performing model.
    """
    print("Loading preprocessed training and testing datasets...")
    if not os.path.exists(config.PROCESSED_TRAIN_PATH) or not os.path.exists(config.PROCESSED_TEST_PATH):
        print("Processed datasets not found! Please run the preprocessing pipeline first.")
        return

    train_df = pd.read_csv(config.PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(config.PROCESSED_TEST_PATH)

    # Split into features (X) and target (y)
    X_train = train_df.drop(columns=["fraud"])
    y_train = train_df["fraud"]
    X_test = test_df.drop(columns=["fraud"])
    y_test = test_df["fraud"]

    print(f"Original training class distribution:\n{y_train.value_counts()}")

    # Apply SMOTE to handle imbalance
    print(f"Applying SMOTE with sampling strategy: {config.SMOTE_SAMPLING_STRATEGY}...")
    smote = SMOTE(sampling_strategy=config.SMOTE_SAMPLING_STRATEGY, random_state=config.RANDOM_SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Resampled training class distribution:\n{y_train_res.value_counts()}")

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            random_state=config.RANDOM_SEED, 
            eval_metric="logloss", 
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            random_state=config.RANDOM_SEED, 
            n_jobs=-1,
            verbose=-1
        )
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining model: {name}...")
        model.fit(X_train_res, y_train_res)
        trained_models[name] = model

        # Predict and calculate probabilities
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON serialization

        results[name] = {
            "Precision": float(precision),
            "Recall": float(recall),
            "F1-Score": float(f1),
            "ROC AUC": float(roc_auc),
            "PR AUC": float(pr_auc),
            "Confusion Matrix": cm
        }

        print(f"{name} Evaluation:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  PR AUC:    {pr_auc:.4f}")

    # Display comparison DataFrame
    comparison_df = pd.DataFrame(results).T.drop(columns=["Confusion Matrix"])
    print("\n--- Model Comparison ---")
    print(comparison_df.to_string())

    # Automatically choose the best model based on F1-Score
    best_model_name = max(results, key=lambda k: results[k]["F1-Score"])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]

    print(f"\nChampion Model: {best_model_name} with F1-Score of {best_metrics['F1-Score']:.4f}")

    # Save best model to destination paths
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    
    # Save the model to src/models as well for dashboard/API fallback imports if configured
    src_best_model_path = os.path.join(config.SRC_MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, src_best_model_path)

    # Save metrics details
    metrics_path = os.path.join(config.MODELS_DIR, "best_model_metrics.json")
    metadata = {
        "model_name": best_model_name,
        "metrics": best_metrics,
        "all_model_results": results
    }
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")
    print(f"Model metrics saved to: {metrics_path}")

if __name__ == "__main__":
    train_and_evaluate_models()
