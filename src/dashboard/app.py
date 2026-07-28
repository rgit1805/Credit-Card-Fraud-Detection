import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import config
from src.dashboard.utils import (
    predict_transaction_api_or_local,
    get_recent_history,
    clear_prediction_logs,
    draw_probability_gauge,
    draw_risk_distribution,
    draw_feature_importance_chart,
    draw_daily_trends,
    draw_prediction_timeline
)
from src.dashboard.styles import inject_styles
from src.explainability.explain import FraudSHAPExplainer
from src.api.predict import preprocess_single, fill_missing_features

# ---------------------------------------------------------
# Page Configuration & Theme Initialization
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fortress Fraud Monitoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_tx_data" not in st.session_state:
    st.session_state.last_tx_data = None

# Sidebar Configuration
st.sidebar.markdown(
    """
    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 20px;'>
        <div style='background: linear-gradient(135deg, #3498DB, #8E44AD); color: white; width: 35px; height: 35px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.2rem;'>F</div>
        <div style='font-size: 1.1rem; font-weight: 700; letter-spacing: 0.5px;'>FORTRESS BANK</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()
st.sidebar.subheader("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home Overview", "Submit Transaction Console", "Historical Logs & Settings"]
)

st.sidebar.divider()
st.sidebar.subheader("Display Settings")
dark_mode = st.sidebar.toggle("Dark Theme Mode", value=True)

# Inject custom styling based on theme mode selection
st.markdown(inject_styles(dark_mode), unsafe_allow_html=True)

# ---------------------------------------------------------
# SHAP Explainer Instance Loader
# ---------------------------------------------------------
@st.cache_resource
def load_shap_explainer():
    try:
        return FraudSHAPExplainer()
    except Exception as e:
        st.sidebar.warning(f"SHAP engine failed to load: {str(e)}")
        return None

shap_explainer = load_shap_explainer()

# Load Champion Model Details
@st.cache_data
def load_model_metrics():
    metrics_path = os.path.join(config.MODELS_DIR, "best_model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

model_metrics = load_model_metrics()

# Fetch recent logs from database
history_df = get_recent_history(1000)

# ---------------------------------------------------------
# Page 1: Home Overview
# ---------------------------------------------------------
if page == "Home Overview":
    st.markdown("<h1 class='header-gradient'>🛡️ Fortress Fraud Control Center</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem;'>Real-time platform oversight, model analytics, and system integrity indicators.</p>", unsafe_allow_html=True)
    st.write("")
    
    # Core Platform Indicators
    total_tx = len(history_df)
    fraud_tx = len(history_df[history_df["prediction"] == 1]) if total_tx > 0 else 0
    fraud_rate = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0.0
    
    # 4-Column Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">MONITORED TRANSACTIONS</div>
                <div class="metric-value">{total_tx:,}</div>
                <div style="font-size:0.75rem; color:#7f8c8d; margin-top:5px;">Total logged in SQLite</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">FRAUD INTERCEPTIONS</div>
                <div class="metric-value" style="color: #e74c3c;">{fraud_tx:,}</div>
                <div style="font-size:0.75rem; color:#7f8c8d; margin-top:5px;">Interdicted transactions</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">DETECTION INCIDENCE RATE</div>
                <div class="metric-value" style="color: { '#e74c3c' if fraud_rate > 1.5 else '#f1c40f' if fraud_rate > 0.5 else '#2ecc71' };">{fraud_rate:.2f}%</div>
                <div style="font-size:0.75rem; color:#7f8c8d; margin-top:5px;">Fraud transaction ratio</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            """
            <div class="glass-card">
                <div class="metric-label">PLATFORM INTEGRITY</div>
                <div class="metric-value" style="color: #2ecc71;">ONLINE</div>
                <div style="font-size:0.75rem; color:#7f8c8d; margin-top:5px;">FastAPI & ML Services healthy</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # Secondary Content Section
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("<h3 class='header-accent'>Activity & Fraud Trends</h3>", unsafe_allow_html=True)
        if not history_df.empty:
            trends_fig = draw_daily_trends(history_df, dark_mode)
            if trends_fig:
                st.plotly_chart(trends_fig, use_container_width=True)
        else:
            st.info("No transaction logs recorded yet. Use the Console tab to submit new transactions.")
            
        # Model Comparison Profile
        st.markdown("<h3 class='header-accent'>Machine Learning Auditing</h3>", unsafe_allow_html=True)
        if model_metrics:
            metrics_df = pd.DataFrame(model_metrics["all_model_results"]).T.drop(columns=["Confusion Matrix"])
            st.dataframe(
                metrics_df.style.highlight_max(axis=0, color="rgba(46, 204, 113, 0.2)"), 
                use_container_width=True
            )
            st.caption(f"Currently serving: **{model_metrics['model_name']}** (auto-selected via cross-validation F1-Score).")
        else:
            st.warning("Model performance logs not found. Run model training scripts first.")

    with col_right:
        st.markdown("<h3 class='header-accent'>Threat Composition</h3>", unsafe_allow_html=True)
        if not history_df.empty:
            dist_fig = draw_risk_distribution(history_df, dark_mode)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
        else:
            st.info("No data available.")
            
        st.markdown("<h3 class='header-accent'>Key Drivers (SHAP)</h3>", unsafe_allow_html=True)
        shap_bar = draw_feature_importance_chart(dark_mode)
        if shap_bar:
            st.plotly_chart(shap_bar, use_container_width=True)
        else:
            st.warning("SHAP feature importance summary CSV not found.")

# ---------------------------------------------------------
# Page 2: Submit Transaction Console
# ---------------------------------------------------------
elif page == "Submit Transaction Console":
    st.markdown("<h1 class='header-gradient'>🎮 Real-Time Risk Analyzer Console</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem;'>Enter realistic transaction parameters below to evaluate fraud risk probabilities instantly.</p>", unsafe_allow_html=True)
    st.write("")
    
    left_col, right_col = st.columns([1.1, 1])
    
    with left_col:
        st.markdown("### Transaction Parameters Input")
        with st.form("transaction_entry_form"):
            # Category 1: Financial Properties
            st.write("**Financial Details**")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                tx_amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=45.50, step=1.0)
                average_tx_amount = st.number_input("Average Monthly Transaction Amount ($)", min_value=0.01, value=55.0, step=1.0)
            with f_col2:
                merchant_category = st.selectbox("Merchant Category", config.MERCHANT_CATEGORIES, index=0)
                payment_method = st.selectbox("Payment Method", config.PAYMENT_METHODS, index=0)

            st.divider()
            
            # Category 2: Transaction Context & Travel
            st.write("**Context & Locations**")
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                merchant_country = st.selectbox("Merchant Country", config.COUNTRIES, index=0)
                distance_from_home = st.number_input("Distance From Customer Home (km)", min_value=0.0, value=8.4)
            with c_col2:
                transaction_type = st.selectbox("Transaction Type", ["purchase", "withdrawal", "transfer"], index=0)
                card_present = st.radio("Physical Card Present?", ["No", "Yes"], index=1, horizontal=True)

            st.divider()
            
            # Category 3: Customer History & Device
            st.write("**Customer Account Profile**")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
                account_age_days = st.number_input("Account Age (Days)", min_value=0, value=730)
                previous_fraud_history = st.radio("Previous Fraud Incidents on Account?", ["No", "Yes"], index=0, horizontal=True)
            with p_col2:
                device_type = st.selectbox("Device Category", config.DEVICE_TYPES, index=0)
                transactions_last_24h = st.number_input("Transactions (Last 24 Hours)", min_value=1, value=2)
                international_transaction = st.radio("Flag as International Transaction?", ["No", "Yes"], index=0, horizontal=True)

            # Conversion checks
            cp_val = 1 if card_present == "Yes" else 0
            pf_val = 1 if previous_fraud_history == "Yes" else 0
            intl_val = 1 if international_transaction == "Yes" else 0

            # Submit
            submit_tx = st.form_submit_button("🔍 Run Prediction Profile")

        # Ingestion Logic
        if submit_tx:
            input_payload = {
                "transaction_amount": tx_amount,
                "merchant_category": merchant_category,
                "merchant_country": merchant_country,
                "payment_method": payment_method,
                "device_type": device_type,
                "transaction_type": transaction_type,
                "customer_age": int(customer_age),
                "account_age_days": int(account_age_days),
                "previous_fraud_history": pf_val,
                "transactions_last_24h": int(transactions_last_24h),
                "average_transaction_amount": average_tx_amount,
                "distance_from_home": distance_from_home,
                "card_present": cp_val,
                "international_transaction": intl_val,
                "customer_id": f"cust_{np.random.randint(1000, 9999)}"
            }
            
            with st.spinner("Processing transaction heuristics and running model inference..."):
                pred_result = predict_transaction_api_or_local(input_payload)
                
            if "error" in pred_result:
                st.error(pred_result["error"])
            else:
                st.session_state.last_prediction = pred_result
                st.session_state.last_tx_data = input_payload
                st.success("Analysis complete!")
                
    with right_col:
        st.markdown("### Risk Evaluation Profile")
        
        if st.session_state.last_prediction is not None:
            res = st.session_state.last_prediction
            tx = st.session_state.last_tx_data
            
            prob = res["fraud_probability"]
            risk = res["risk_level"]
            action = res["recommended_action"]
            conf = (1 - prob) * 100 if res["prediction"] == 0 else prob * 100
            
            # Risk Level Badge Box
            risk_class = "status-low" if risk == "LOW" else "status-medium" if risk == "MEDIUM" else "status-high"
            action_class = "action-approve" if action == config.ACTION_APPROVE else "action-otp" if action == config.ACTION_OTP else "action-review" if action == config.ACTION_REVIEW else "action-block"
            
            # Gauge rendering
            gauge_fig = draw_probability_gauge(prob, dark_mode)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Info Grid
            c_badge, c_conf = st.columns(2)
            with c_badge:
                st.markdown(
                    f"""
                    <div style='text-align: center;'>
                        <div class="metric-label">RISK LEVEL</div>
                        <div class="status-badge {risk_class}" style="font-size: 1.1rem; width:80%;">{risk}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with c_conf:
                st.markdown(
                    f"""
                    <div style='text-align: center;'>
                        <div class="metric-label">EVALUATION CONFIDENCE</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #3498DB;">{conf:.2f}%</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
            st.write("")
            
            # Policy Decision Box
            st.markdown(
                f"""
                <div class="action-box {action_class}">
                    <div style="font-size:0.8rem; text-transform:uppercase; margin-bottom:5px;">POLICY DECISION REQUIREMENT</div>
                    <div style="font-size:1.6rem; font-weight:700;">{action}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.write("")
            
            # SHAP Explainability Sub-Panel
            st.markdown("#### Explain prediction details (Local SHAP values)")
            if shap_explainer:
                try:
                    # Retrieve OHE preprocessed format of the filled record
                    filled_record = res.get("filled_transaction")
                    if not filled_record:
                        filled_record = fill_missing_features(tx)
                        
                    processed_df = preprocess_single(filled_record)
                    shap_values = shap_explainer.explain_transaction(processed_df)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    # Adjust text sizes and color configurations based on theme
                    plt.rcParams.update({
                        'text.color': 'white' if dark_mode else 'black',
                        'axes.labelcolor': 'white' if dark_mode else 'black',
                        'xtick.color': 'white' if dark_mode else 'black',
                        'ytick.color': 'white' if dark_mode else 'black'
                    })
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    
                    # Generate waterfall
                    shap.plots.waterfall(shap_values[0], max_display=7, show=False)
                    plt.title("Prediction Decision Factors Impact", fontsize=11, color='white' if dark_mode else 'black')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as shap_err:
                    st.warning(f"Could not render explanation factors: {str(shap_err)}")
            else:
                st.info("SHAP engine not active. Global summary weights available on the Home page.")
                
        else:
            st.markdown(
                """
                <div style='text-align: center; border: 2px dashed rgba(128,128,128,0.2); border-radius: 12px; padding: 50px; margin-top: 30px;'>
                    <div style='font-size: 3rem; color: rgba(128,128,128,0.3);'>🛡️</div>
                    <div style='font-size: 1.1rem; color: gray; margin-top: 10px;'>Awaiting console input profile submission...</div>
                </div>
                """, 
                unsafe_allow_html=True
            )

# ---------------------------------------------------------
# Page 3: Historical Logs & Settings
# ---------------------------------------------------------
elif page == "Historical Logs & Settings":
    st.markdown("<h1 class='header-gradient'>📜 Transaction Registry & Logs</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem;'>Audit past risk decisions, download history as CSV, or clear prediction registries.</p>", unsafe_allow_html=True)
    st.write("")
    
    if not history_df.empty:
        # Timeline visualizer
        st.markdown("### Transaction Ingestion Timeline")
        timeline_fig = draw_prediction_timeline(history_df, dark_mode)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
            
        st.write("")
        
        # Grid Controls
        col_grid, col_actions = st.columns([4, 1])
        with col_grid:
            st.markdown("### Decision Registry Table")
            
            # Format and display cleaner grid view
            display_df = history_df[[
                "id", "timestamp", "customer_id", "transaction_amount", 
                "merchant_category", "merchant_country", "payment_method",
                "fraud_probability", "risk_level", "recommended_action", "prediction"
            ]].copy()
            
            display_df["prediction"] = display_df["prediction"].map({0: "Approved", 1: "Blocked"})
            display_df["fraud_probability"] = (display_df["fraud_probability"] * 100).round(2).astype(str) + "%"
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
        with col_actions:
            st.markdown("### Audit Actions")
            
            # Download CSV
            csv_data = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Logs as CSV",
                data=csv_data,
                file_name=f"fraud_predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.write("")
            st.write("")
            
            # Settings options
            st.markdown("**System Management**")
            if st.button("⚠️ Clear Predictions History", use_container_width=True, type="secondary"):
                clear_prediction_logs()
                st.success("Prediction logs cleared! Reloading...")
                st.rerun()
    else:
        st.info("No transaction registry logs available. Submit new transaction profiles in the Console tab.")