import os
import sys
import requests
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import config
from src.database.db_manager import DatabaseManager

API_URL = "http://127.0.0.1:8000"

# Initialize DB Manager
db = DatabaseManager()

def predict_transaction_api_or_local(data: dict) -> dict:
    """
    Predicts a transaction by querying the FastAPI backend.
    If the API server is offline or fails, falls back gracefully to local ML execution
    and logs the transaction directly to SQLite.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=1.0)
        if response.status_code == 200:
            pred_response = requests.post(f"{API_URL}/predict", json=data, timeout=3.0)
            if pred_response.status_code == 200:
                # Return API results
                return pred_response.json()
    except Exception as e:
        # Log to stdout and proceed to fallback
        print(f"FastAPI backend connection error: {str(e)}. Using local fallback...")
        
    # Local fallback
    try:
        # Delay import to prevent circular references/dependency constraints during setup
        from src.api.predict import predict_transaction
        
        pred_res = predict_transaction(data)
        # Log local prediction to SQLite
        db.log_prediction(pred_res["filled_transaction"], pred_res)
        
        return {
            "prediction": pred_res["prediction"],
            "fraud_probability": pred_res["fraud_probability"],
            "risk_level": pred_res["risk_level"],
            "recommended_action": pred_res["recommended_action"],
            "fallback_used": True
        }
    except Exception as local_err:
        print(f"CRITICAL: Local fallback prediction failed: {str(local_err)}")
        return {"error": f"Inference engine failure: {str(local_err)}"}

def get_recent_history(limit: int = 1000) -> pd.DataFrame:
    """
    Fetches prediction histories from database.
    """
    return db.get_history_df(limit)

def clear_prediction_logs():
    """
    Deletes history from database.
    """
    db.clear_history()

def draw_probability_gauge(prob: float, is_dark: bool = True):
    """
    Draws a responsive Plotly circular gauge showing risk regions.
    """
    text_color = "#ECF0F1" if is_dark else "#2C3E50"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fraud Probability (%)", 'font': {'size': 16, 'color': text_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': text_color, 'tickfont': {'color': text_color}},
            'bar': {'color': "#34495E" if is_dark else "#BDC3C7"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1.5,
            'bordercolor': "rgba(128,128,128,0.5)",
            'steps': [
                {'range': [0, config.RISK_THRESHOLD_LOW * 100], 'color': 'rgba(46, 204, 113, 0.25)'},
                {'range': [config.RISK_THRESHOLD_LOW * 100, config.RISK_THRESHOLD_MEDIUM * 100], 'color': 'rgba(241, 196, 15, 0.25)'},
                {'range': [config.RISK_THRESHOLD_MEDIUM * 100, 100], 'color': 'rgba(231, 76, 60, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#E74C3C", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color, 'family': "Inter"},
        height=220,
        margin=dict(l=25, r=25, t=40, b=15)
    )
    return fig

def draw_risk_distribution(df: pd.DataFrame, is_dark: bool = True):
    """
    Renders a donut chart illustrating LOW, MEDIUM, and HIGH risk volume shares.
    """
    if df.empty:
        return None
        
    counts = df["risk_level"].value_counts().reset_index()
    counts.columns = ["Risk Level", "Count"]
    
    color_map = {
        "LOW": "#2ECC71",
        "MEDIUM": "#F1C40F",
        "HIGH": "#E74C3C"
    }
    
    fig = px.pie(
        counts, 
        values="Count", 
        names="Risk Level",
        color="Risk Level",
        color_discrete_map=color_map,
        hole=0.45
    )
    
    text_color = "#ECF0F1" if is_dark else "#2C3E50"
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color, 'family': "Inter"},
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def draw_feature_importance_chart(is_dark: bool = True):
    """
    Loads pre-computed SHAP importances and draws a horizontal bar chart.
    """
    if not os.path.exists(config.SHAP_SUMMARY_PATH):
        return None
        
    df = pd.read_csv(config.SHAP_SUMMARY_PATH).head(10)
    # Sort for horizontal visualization
    df = df.sort_values(by="importance", ascending=True)
    
    fig = px.bar(
        df, 
        x="importance", 
        y="feature", 
        orientation='h',
        color_discrete_sequence=["#3498DB"]
    )
    
    text_color = "#ECF0F1" if is_dark else "#2C3E50"
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color, 'family': "Inter"},
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Average Absolute SHAP Impact",
        yaxis_title=""
    )
    return fig

def draw_daily_trends(df: pd.DataFrame, is_dark: bool = True):
    """
    Plots a dual-axis graph displaying total volumes vs fraudulent detections over days.
    """
    if df.empty:
        return None
        
    df_copy = df.copy()
    # Normalize timestamp formats
    df_copy["date"] = pd.to_datetime(df_copy["timestamp"]).dt.strftime("%Y-%m-%d")
    
    daily = df_copy.groupby("date").agg(
        total_volume=("id", "count"),
        fraud_volume=("prediction", "sum")
    ).reset_index()
    
    # Sort chronological
    daily = daily.sort_values(by="date")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily["date"],
        y=daily["total_volume"],
        name="Total Transactions",
        marker_color="rgba(52, 152, 219, 0.45)" if is_dark else "rgba(52, 152, 219, 0.7)",
        yaxis="y"
    ))
    
    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["fraud_volume"],
        name="Fraudulent Detections",
        line=dict(color="#E74C3C", width=3),
        mode="lines+markers",
        yaxis="y2"
    ))
    
    text_color = "#ECF0F1" if is_dark else "#2C3E50"
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color, 'family': "Inter"},
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        yaxis=dict(
            title=dict(
        text="Total Transactions",
        font=dict(color="#3498DB")
    ),
    tickfont=dict(color="#3498DB"),
    gridcolor="rgba(128,128,128,0.15)"
),
        yaxis2=dict(
    title=dict(
        text="Fraud Detections",
        font=dict(color="#E74C3C")
    ),
    tickfont=dict(color="#E74C3C"),
    overlaying="y",
    side="right"
)
    )
    return fig

def draw_prediction_timeline(df: pd.DataFrame, is_dark: bool = True):
    """
    Renders a timeline highlighting transactional spikes and fraud points.
    """
    if df.empty:
        return None
        
    df_copy = df.copy().head(50)  # limit to recent 50
    df_copy["color"] = df_copy["prediction"].map({0: "#2ECC71", 1: "#E74C3C"})
    df_copy["size"] = df_copy["transaction_amount"].apply(lambda x: min(x / 5.0 + 8.0, 45.0))
    
    text_color = "#ECF0F1" if is_dark else "#2C3E50"
    
    fig = go.Figure()
    
    # Split genuine and fraud for cleaner legends
    genuine = df_copy[df_copy["prediction"] == 0]
    fraud = df_copy[df_copy["prediction"] == 1]
    
    if not genuine.empty:
        fig.add_trace(go.Scatter(
            x=genuine["timestamp"],
            y=genuine["transaction_amount"],
            mode="markers",
            name="Approved",
            marker=dict(
                color="#2ECC71",
                size=genuine["size"],
                line=dict(width=1, color="white" if is_dark else "black")
            ),
            text="Customer ID: " + genuine["customer_id"] + "<br>Amount: $" + genuine["transaction_amount"].astype(str)
        ))
        
    if not fraud.empty:
        fig.add_trace(go.Scatter(
            x=fraud["timestamp"],
            y=fraud["transaction_amount"],
            mode="markers",
            name="Blocked (Fraud)",
            marker=dict(
                color="#E74C3C",
                size=fraud["size"],
                symbol="x",
                line=dict(width=1, color="white" if is_dark else "black")
            ),
            text="Customer ID: " + fraud["customer_id"] + "<br>Amount: $" + fraud["transaction_amount"].astype(str)
        ))
        
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color, 'family': "Inter"},
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Timestamp",
        yaxis_title="Transaction Amount ($)",
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(gridcolor="rgba(128,128,128,0.05)")
    )
    return fig
