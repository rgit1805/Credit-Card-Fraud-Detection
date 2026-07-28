import os
import sys
import joblib
import pandas as pd
import numpy as np
import shap

# Add src to python path to import configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import config

class FraudSHAPExplainer:
    """
    Handles SHAP explainability calculations for the selected champion model,
    providing individual and global prediction explanations.
    """
    def __init__(self):
        # Load best model
        if not os.path.exists(config.BEST_MODEL_PATH):
            raise FileNotFoundError(f"Champion model not found at {config.BEST_MODEL_PATH}. Train models first.")
        
        self.model = joblib.load(config.BEST_MODEL_PATH)
        
        # Load background training data to initialize explainer
        if os.path.exists(config.PROCESSED_TRAIN_PATH):
            train_df = pd.read_csv(config.PROCESSED_TRAIN_PATH)
            X_train = train_df.drop(columns=["fraud"])
            # Sample background data for baseline references (e.g. 100 samples)
            self.background_data = shap.sample(X_train, 100, random_state=config.RANDOM_SEED)
        else:
            self.background_data = None
            
        # Determine appropriate explainer type
        model_type_str = str(type(self.model)).lower()
        if "forest" in model_type_str or "xgb" in model_type_str or "lgbm" in model_type_str:
            print("Tree-based model detected. Initializing SHAP TreeExplainer...")
            # TreeExplainer is highly optimized for tree models
            self.explainer = shap.TreeExplainer(self.model)
        else:
            print("Non-tree model detected. Initializing general SHAP Explainer...")
            self.explainer = shap.Explainer(self.model, self.background_data)

    def explain_transaction(self, preprocessed_df: pd.DataFrame) -> shap.Explanation:
        """
        Computes SHAP values for a single transaction dataframe.
        """
        shap_values = self.explainer(preprocessed_df)
        return shap_values

    def compute_and_save_global_importance(self, output_path: str = config.SHAP_SUMMARY_PATH):
        """
        Computes global SHAP feature importance on a sample of data and saves it to a CSV.
        Optimizes dashboard visualization rendering.
        """
        if self.background_data is None:
            print("Training data not available. Cannot calculate global SHAP importance.")
            return

        print("Computing global SHAP feature importance...")
        shap_values = self.explainer(self.background_data)
        
        # Extract mean absolute SHAP values per feature
        if isinstance(shap_values, shap.Explanation):
            # In SHAP 0.40+, explainer(data) returns an Explanation object
            # Note: For multi-class or binary classification, tree explainer might output 3D values (nsamples, nfeatures, nclasses) 
            # or 2D values (nsamples, nfeatures). For scikit-learn RandomForest, shap_values.values is often (nsamples, nfeatures, 2)
            vals = shap_values.values
            if len(vals.shape) == 3:  # (samples, features, classes)
                # Take positive class index 1
                mean_shap = np.abs(vals[:, :, 1]).mean(axis=0)
            else:
                mean_shap = np.abs(vals).mean(axis=0)
                
            features = shap_values.feature_names
        else:
            # Fallback for old SHAP versions
            if isinstance(shap_values, list):
                # Class 1 (fraud)
                mean_shap = np.abs(shap_values[1]).mean(axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
            features = self.background_data.columns.tolist()

        importance_df = pd.DataFrame({
            "feature": features,
            "importance": mean_shap
        }).sort_values(by="importance", ascending=False)
        
        # Save to file
        importance_df.to_csv(output_path, index=False)
        print(f"Global SHAP feature importance saved to: {output_path}")
        return importance_df

if __name__ == "__main__":
    # Precompute global SHAP importance
    explainer = FraudSHAPExplainer()
    explainer.compute_and_save_global_importance()
