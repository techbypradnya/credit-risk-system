# Name : Pradnya Shailendra Pangavahne
# Roll No : 43
# PRN No  : UEC23F043


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.predict import predict_loan, risk_category

# Generate sample data for dashboard
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 5000
    data = {
        'person_age': np.random.normal(40, 12, n).clip(18, 75).astype(int),
        'person_income': np.random.lognormal(11, 0.6, n).clip(10000, 2000000).astype(int),
        'loan_amnt': np.random.normal(100000, 50000, n).clip(5000, 500000).astype(int),
        'loan_int_rate': np.random.normal(12, 3, n).clip(5, 25),
        'loan_percent_income': np.random.uniform(0.05, 0.45, n),
        'person_emp_length': np.random.gamma(5, 2, n).clip(0, 40).astype(int),
        'cb_person_cred_hist_length': np.random.gamma(8, 1.5, n).clip(1, 30).astype(int),
        'person_home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], n),
        'loan_grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n),
        'loan_intent': np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT'], n),
        'cb_person_default_on_file': np.random.choice(['N', 'Y'], n, p=[0.92, 0.08])
    }
    df = pd.DataFrame(data)
    
    # Realistic risk scoring
    df['risk_score'] = (
        0.3 * (df['loan_percent_income'] > 0.35) +
        0.25 * (df['person_age'] < 30) +
        0.2 * (df['loan_int_rate'] > 15) +
        0.15 * (df['cb_person_default_on_file'] == 'Y') +
        0.1 * (df['person_emp_length'] < 3)
    )
    df['risk'] = pd.cut(df['risk_score'], [0, 0.3, 0.6, 1], 
                       labels=['Low Risk', 'Medium Risk', 'High Risk'])
    df['income'] = df['person_income']
    df['loan_amount'] = df['loan_amnt']
    return df

df = load_sample_data()

st.set_page_config(page_title="Credit Risk System", page_icon="💳", layout="wide")

# ─── GLOBAL CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* Dark background */
.stApp {
    background: #0F172A !important;
    color: #F1F5F9 !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

/* ── All labels bright & visible ── */
label, .stSlider label, .stNumberInput label,
.stSelectbox label, [data-testid="stWidgetLabel"] {
    color: #CBD5E1 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}

/* ── Input fields ── */
input[type="number"], .stNumberInput input {
    background: #0F172A !important;
    color: #F1F5F9 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
input[type="number"]:focus {
    border-color: #378ADD !important;
    box-shadow: 0 0 0 2px rgba(55,138,221,0.25) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    font-weight: 500 !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #60A5FA !important;
    border: 2px solid #0F172A !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,.3) !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #1D4ED8, #7C3AED) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    height: 52px !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
div[data-testid="stButton"] > button:hover { opacity: .88 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #94A3B8 !important;
    text-transform: uppercase !important;
    letter-spacing: .07em !important;
}
[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #16A34A, #4ADE80) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div style="
    text-align:center;
    padding:2rem 1.5rem 1.75rem;
    background:linear-gradient(135deg,#1E3A5F 0%,#1a2a4a 100%);
    border-radius:18px;
    border:1px solid #2D4A6E;
    margin-bottom:1.5rem;
">
    <div style="display:inline-block;background:#378ADD;color:#fff;font-size:11px;font-weight:700;
        padding:4px 16px;border-radius:99px;letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px;">
        AI-Powered · Credit Analysis
    </div>
    <div style="font-size:30px;font-weight:700;color:#FFFFFF;margin-bottom:8px;text-shadow:0 1px 6px rgba(0,0,0,.4);">
        Credit Risk Prediction System
    </div>
    <div style="font-size:15px;color:#94BFDF;">
        Enter applicant details to evaluate loan default probability
    </div>
</div>
""", unsafe_allow_html=True)

# ─── SECTION HEADER HELPER ─────────────────────────────────────
def section(title, color, dot_color):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;margin-top:.25rem;">
        <div style="width:10px;height:10px;border-radius:50%;background:{dot_color};flex-shrink:0;
            box-shadow:0 0 8px {dot_color}88;"></div>
        <span style="font-size:12px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:{color};">
            {title}
        </span>
    </div>
    """, unsafe_allow_html=True)

# ─── CARD 1: PERSONAL INFO ─────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    with st.container():
        st.markdown('<div style="background:#1E293B;border:1px solid #2D3F55;border-radius:16px;padding:1.5rem;">', unsafe_allow_html=True)
        section("Personal Information", "#60A5FA", "#378ADD")

        c1, c2, c3 = st.columns(3)
        with c1: age = st.slider("Age", 18, 100, 30)
        with c2: income = st.number_input("Annual Income (₹)", min_value=0, step=10000, value=500000)
        with c3: emp_length = st.slider("Employment Length (yrs)", 0, 40, 5)

        c4, c5 = st.columns(2)
        with c4: home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        with c5: cred_hist = st.slider("Credit History Length (yrs)", 0, 30, 10)

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ─── CARD 2: LOAN DETAILS ──────────────────────────────────────
with st.container():
    st.markdown('<div style="background:#1E293B;border:1px solid #2D3F55;border-radius:16px;padding:1.5rem;">', unsafe_allow_html=True)
    section("Loan Details", "#A78BFA", "#7C3AED")

    c6, c7, c8 = st.columns(3)
    with c6: loan = st.number_input("Loan Amount (₹)", min_value=0, step=5000, value=100000)
    with c7: interest = st.number_input("Interest Rate (%)", min_value=0.0, step=0.5, format="%.1f", value=12.0)
    with c8: percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=100.0, step=0.5, format="%.1f", value=20.0)

    c9, c10, c11 = st.columns(3)
    with c9: intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"])
    with c10: grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    with c11: default_hist = st.selectbox("Previous Default", ["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

# ─── PREDICT BUTTON ────────────────────────────────────────────
predict = st.button("🔍  Predict Risk", use_container_width=True)

# ─── PREPROCESSING ─────────────────────────────────────────────
def preprocess_input():
    return {
        'person_age': age, 'person_income': income,
        'person_home_ownership': home, 'person_emp_length': emp_length,
        'loan_intent': intent, 'loan_grade': grade,
        'loan_amnt': loan, 'loan_int_rate': interest,
        'loan_percent_income': percent_income,
        'cb_person_default_on_file': default_hist,
        'cb_person_cred_hist_length': cred_hist
    }
# ─── RESULTS ───────────────────────────────────────────────────
if predict:
    with st.spinner("Analyzing applicant profile..."):
        try:
            # ✅ SINGLE SOURCE OF TRUTH
            risk, prob = predict_loan(preprocess_input())

        except Exception as e:
            st.error("Prediction error. Please check input.")
            risk = "Medium Risk"
            prob = 0.5

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div style="background:linear-gradient(135deg,#1E293B,#1a2540);border:1px solid #334155;border-radius:16px;padding:1.5rem;">', unsafe_allow_html=True)
    section("Prediction Result", "#2DD4BF", "#0D9488")

    m1, m2, m3 = st.columns(3)

    # ✅ Prediction (ONLY risk, no binary confusion)
    with m1:
        st.metric(
            "Prediction",
            risk
        )

    # ✅ Risk Score
    with m2:
        st.metric("Risk Score", f"{prob*100:.1f}%")

    # ✅ Risk Level
    with m3:
        st.metric("Risk Level", risk)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ✅ Progress Bar Label
    col_l, col_r = st.columns([5, 1])
    with col_l:
        st.caption("Default probability")
    with col_r:
        pct_color = "#4ADE80" if "Low" in risk else ("#FBBF24" if "Medium" in risk else "#F87171")
        st.markdown(f"<p style='text-align:right;font-size:12px;font-weight:700;color:{pct_color};margin:0'>{prob*100:.1f}%</p>", unsafe_allow_html=True)

    st.progress(int(prob * 100))

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ✅ Alert
    if "High" in risk:
        st.error("🚨 High risk applicant — loan approval is not recommended.")
    elif "Medium" in risk:
        st.warning("⚠️ Medium risk — further verification is recommended.")
    else:
        st.success("✅ Low risk applicant — safe for loan approval.")

    # ✅ Insight
    if "High" in risk:
        msg = "High default probability detected. Review income stability, credit history, and consider collateral before approval."
        accent = "#F87171"
    elif "Medium" in risk:
        msg = "Moderate risk detected. Additional document verification and collateral assessment is advised."
        accent = "#FBBF24"
    else:
        msg = "Strong financial profile. Income-to-loan ratio and credit history are within safe thresholds. Approval recommended."
        accent = "#2DD4BF"

    st.markdown(f"""
    <div style="background:#0F172A;border-left:3px solid {accent};border-radius:0 10px 10px 0;
        padding:12px 16px;font-size:13px;color:#CBD5E1;line-height:1.7;margin-top:.5rem;">
        <strong style="color:{accent};">Insight: </strong>{msg}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Charts Row 2
col_scatter1, col_scatter2 = st.columns(2)

with col_scatter1:
    section("Income vs Loan Amount", "#3B82F6", "#1D4ED8")
    fig_scatter = px.scatter(df.sample(2000), x='income', y='loan_amount', 
                           color='risk',
                           color_discrete_map={'Low Risk': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'},
                           log_x=True, height=350)
    fig_scatter.update_traces(marker=dict(size=6))
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_scatter2:
    section("DTI vs Default Risk", "#F59E0B", "#D97706")
    df_sample = df.sample(2000)
    df_sample['default_risk'] = df_sample['risk_score']
    fig_dti = px.scatter(df_sample, x='loan_percent_income', y='default_risk',
                        color='risk',
                        size='loan_amount',
                        color_discrete_map={'Low Risk': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#EF4444'},
                        labels={'loan_percent_income': 'DTI Ratio', 'default_risk': 'Risk Score'},
                        height=350)
    st.plotly_chart(fig_dti, use_container_width=True)

# Correlation Heatmap
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
section("Key Risk Factor Correlation", "#EC4899", "#BE185D")

numeric_cols = ['person_age', 'person_income', 'loan_amount', 'loan_int_rate', 
               'loan_percent_income', 'person_emp_length', 'cb_person_cred_hist_length']
corr = df[numeric_cols].corr()
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", height=400)
fig_corr.update_layout(margin=dict(t=40, b=0))
st.plotly_chart(fig_corr, use_container_width=True)

# Portfolio Risk Gauge
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
col_gauge1, col_gauge2 = st.columns([3, 1])

with col_gauge1:
    section("Portfolio Risk Overview", "#EF4444", "#DC2626")
    
with col_gauge2:
    risk_pct = (df['risk'] == 'High Risk').mean() * 100
    st.metric("High Risk %", f"{risk_pct:.1f}%")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_pct,
    title={'text': "Portfolio Risk"},
    gauge={'axis': {'range': [0, 50]},
           'bar': {'color': "#EF4444"},
           'steps': [
               {'range': [0, 15], 'color': '#10B981'},
               {'range': [15, 30], 'color': '#F59E0B'},
               {'range': [30, 50], 'color': '#EF4444'}
           ],
           'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': risk_pct}}
))
fig_gauge.update_layout(height=300)
st.plotly_chart(fig_gauge, use_container_width=True)