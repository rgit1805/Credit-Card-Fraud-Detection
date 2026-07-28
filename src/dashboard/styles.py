# Custom CSS styles for premium, responsive UI in Streamlit

CSS_BASE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
    transition: all 0.3s ease-in-out;
}

.glass-card:hover {
    border-color: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
}

/* Light Mode Override for Glassmorphism */
.light-mode .glass-card {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(0, 0, 0, 0.08);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
}

.light-mode .glass-card:hover {
    border-color: rgba(0, 0, 0, 0.15);
    box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.12);
}

/* Value Status Badges */
.status-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    text-align: center;
}

.status-low {
    background-color: rgba(46, 204, 113, 0.15);
    color: #2ecc71;
    border: 1px solid rgba(46, 204, 113, 0.3);
}

.status-medium {
    background-color: rgba(241, 196, 15, 0.15);
    color: #f1c40f;
    border: 1px solid rgba(241, 196, 15, 0.3);
}

.status-high {
    background-color: rgba(231, 76, 60, 0.15);
    color: #e74c3c;
    border: 1px solid rgba(231, 76, 60, 0.3);
}

/* System Action Banners */
.action-box {
    border-radius: 12px;
    padding: 20px;
    margin-top: 15px;
    font-weight: 500;
}

.action-approve {
    background: linear-gradient(135deg, rgba(39,174,96,0.1), rgba(46,204,113,0.2));
    border: 1.5px solid #2ecc71;
    color: #2ecc71;
}

.action-otp {
    background: linear-gradient(135deg, rgba(230,126,34,0.1), rgba(241,196,15,0.2));
    border: 1.5px solid #f39c12;
    color: #f39c12;
}

.action-review {
    background: linear-gradient(135deg, rgba(155,89,182,0.1), rgba(142,68,173,0.2));
    border: 1.5px solid #9b59b6;
    color: #9b59b6;
}

.action-block {
    background: linear-gradient(135deg, rgba(192,57,43,0.1), rgba(231,76,60,0.2));
    border: 1.5px solid #e74c3c;
    color: #e74c3c;
}

/* Header Text styling */
.header-gradient {
    background: linear-gradient(90deg, #3498db, #9b59b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

.header-accent {
    font-weight: 700;
    color: #3498db;
}

/* Bank Branding Widget */
.bank-brand {
    display: flex;
    align-items: center;
    gap: 15px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    margin-bottom: 25px;
}

.bank-logo {
    background: linear-gradient(135deg, #3498db, #8e44ad);
    color: white;
    width: 45px;
    height: 45px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1.5rem;
}

.bank-name {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* Custom Metric Layout */
.metric-container {
    display: flex;
    justify-content: space-between;
    gap: 15px;
}

.metric-item {
    flex: 1;
    text-align: center;
    background: rgba(128,128,128,0.05);
    border-radius: 8px;
    padding: 12px;
}

.metric-label {
    font-size: 0.8rem;
    color: #7f8c8d;
    margin-bottom: 5px;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 600;
}
</style>
"""

def inject_styles(is_dark: bool = True):
    """
    Returns custom CSS code wrapper to inject into Streamlit dashboard.
    """
    mode_class = "dark-mode" if is_dark else "light-mode"
    styled_html = CSS_BASE.replace("<style>", f"<style>\nbody {{ content: '{mode_class}'; }}")
    # Add wrapper class div depending on mode selection
    return styled_html
