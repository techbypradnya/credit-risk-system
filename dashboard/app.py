import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.predict import predict_loan, risk_category

st.set_page_config(page_title="Credit Risk System", page_icon="💳", layout="centered")

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
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 860px; }

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
.stSelectbox svg { fill: #64748B !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #60A5FA !important;
    border: 2px solid #0F172A !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,.3) !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stTickBarMin"],
.stSlider [data-baseweb="slider"] div[data-testid="stTickBarMax"] {
    color: #64748B !important;
    font-size: 11px !important;
}
.stSlider [data-baseweb="slider"] > div > div:first-child {
    background: #1E3A5F !important;
}
.stSlider [data-baseweb="slider"] > div > div:nth-child(2) {
    background: linear-gradient(90deg, #1D4ED8, #60A5FA) !important;
}

/* ── Slider value pill ── */
.stSlider [data-testid="stThumbValue"] {
    background: #1D4ED8 !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    border-radius: 6px !important;
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
    transition: opacity .15s !important;
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
[data-testid="stMetricDelta"] {
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div {
    background: #1E293B !important;
    border-radius: 4px !important;
    height: 8px !important;
    border: 1px solid #334155 !important;
}
[data-testid="stProgress"] > div > div {
    border-radius: 4px !important;
    height: 8px !important;
    background: linear-gradient(90deg, #16A34A, #4ADE80) !important;
}

/* ── Alert/Success/Warning/Error boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* ── Caption / small text ── */
.stCaption, small {
    color: #94A3B8 !important;
    font-size: 12px !important;
}

/* ── Divider ── */
hr { border-color: #1E293B !important; margin: 1.5rem 0 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: #94A3B8 !important; }
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
with st.container():
    st.markdown('<div style="background:#1E293B;border:1px solid #2D3F55;border-radius:16px;padding:1.5rem;">', unsafe_allow_html=True)
    section("Personal Information", "#60A5FA", "#378ADD")

    c1, c2, c3 = st.columns(3)
    with c1: age = st.slider("Age", 18, 100, 30)
    with c2: income = st.number_input("Annual Income (₹)", min_value=0, step=10000)
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
    with c6: loan = st.number_input("Loan Amount (₹)", min_value=0, step=5000)
    with c7: interest = st.number_input("Interest Rate (%)", min_value=0.0, step=0.5, format="%.1f")
    with c8: percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=100.0, step=0.5, format="%.1f")

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
        pred, prob = predict_loan(preprocess_input())
        risk = risk_category(prob)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div style="background:linear-gradient(135deg,#1E293B,#1a2540);border:1px solid #334155;border-radius:16px;padding:1.5rem;">', unsafe_allow_html=True)
    section("Prediction Result", "#2DD4BF", "#0D9488")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Prediction", "Default ✗" if pred == 1 else "Safe ✓",
                  delta="High Risk" if pred == 1 else "Low Risk", delta_color="inverse")
    with m2:
        st.metric("Risk Score", f"{prob:.2f}")
    with m3:
        st.metric("Risk Level", risk)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # Labeled progress bar
    col_l, col_r = st.columns([5, 1])
    with col_l:
        st.caption("Default probability")
    with col_r:
        pct_color = "#4ADE80" if prob < 0.3 else ("#FBBF24" if prob < 0.7 else "#F87171")
        st.markdown(f"<p style='text-align:right;font-size:12px;font-weight:700;color:{pct_color};margin:0'>{prob*100:.1f}%</p>", unsafe_allow_html=True)
    st.progress(int(prob * 100))

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # Alert
    if risk == "High Risk":
        st.error("🚨 High risk applicant — loan approval is not recommended.")
    elif risk == "Medium Risk":
        st.warning("⚠️ Medium risk — further verification is recommended.")
    else:
        st.success("✅ Low risk applicant — safe for loan approval.")

    # Insight box
    if prob > 0.7:
        msg = "High default probability detected. Review income stability, credit history, and consider collateral before approval."
        accent = "#F87171"
    elif prob > 0.3:
        msg = "Moderate risk detected. Additional document verification and a collateral assessment is advised."
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