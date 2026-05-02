import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

st.set_page_config(
    page_title="Marketing Campaign Predictor",
    page_icon="📊",
    layout="wide"
)


PARAM_DIR = "parameters"
PARAM_DIR = os.path.join("model", "parameters")

@st.cache_resource
def load_artifacts():
    reg_model   = pickle.load(open(os.path.join(PARAM_DIR, "regression_model.pkl"),        "rb"))
    clf_model   = pickle.load(open(os.path.join(PARAM_DIR, "classification_model.pkl"),    "rb"))
    reg_feats   = pickle.load(open(os.path.join(PARAM_DIR, "regression_features.pkl"),     "rb"))
    clf_feats   = pickle.load(open(os.path.join(PARAM_DIR, "classification_features.pkl"), "rb"))
    mappings    = json.load(open(os.path.join(PARAM_DIR,   "label_mappings.json")))
    return reg_model, clf_model, reg_feats, clf_feats, mappings

reg_model, clf_model, reg_feats, clf_feats, mappings = load_artifacts()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📊 Marketing Campaign Performance Predictor")
st.markdown("Predict **Revenue** and **Profit / Loss** for a campaign using trained ML models.")
st.divider()

# ─────────────────────────────────────────────
# Sidebar — Campaign Inputs
# ─────────────────────────────────────────────
st.sidebar.header("🎯 Campaign Details")
st.sidebar.markdown("Fill in the campaign parameters below.")

# --- Categorical Inputs ---
campaign_type = st.sidebar.selectbox(
    "Campaign Type",
    options=list(mappings["Campaign_Type"].keys()),
    format_func=lambda x: x.title()
)

target_audience = st.sidebar.selectbox(
    "Target Audience",
    options=list(mappings["Target_Audience"].keys()),
    format_func=lambda x: x.title()
)

customer_segment = st.sidebar.selectbox(
    "Customer Segment",
    options=list(mappings["Customer_Segment"].keys()),
    format_func=lambda x: x.title()
)

# --- Channel Selection ---
st.sidebar.markdown("**Channels Used**")
all_channels = ["email", "facebook", "google", "instagram", "whatsapp", "youtube"]
selected_channels = []
cols = st.sidebar.columns(2)
for i, ch in enumerate(all_channels):
    if cols[i % 2].checkbox(ch.title(), value=(ch == "facebook")):
        selected_channels.append(ch)

st.sidebar.divider()

# --- Numeric Inputs ---
duration      = st.sidebar.slider("Duration (days)",         1,   90,  15)
impressions   = st.sidebar.number_input("Impressions",        1000, 500000, 30000, step=1000)
clicks        = st.sidebar.number_input("Clicks",             100,  50000,  3000,  step=100)
leads         = st.sidebar.number_input("Leads",              10,   20000,  1000,  step=50)
conversions   = st.sidebar.number_input("Conversions",        1,    10000,  500,   step=50)
acquisition_cost = st.sidebar.number_input("Acquisition Cost (₹)", 10.0, 1000.0, 150.0, step=10.0)
engagement_score = st.sidebar.slider("Engagement Score",     1,   100,  50)

# Total Cost derived
total_cost = acquisition_cost * clicks
st.sidebar.metric("Total Cost (auto-calculated)", f"₹ {total_cost:,.0f}")

# ─────────────────────────────────────────────
# Feature Engineering — mirror training pipeline
# ─────────────────────────────────────────────
def build_input():
    # Log transforms
    conversions_log      = np.log(conversions + 1)
    acquisition_cost_log = np.log(acquisition_cost + 1)

    # Interaction features (regression only)
    conv_per_click        = conversions / (clicks + 1)
    lead_per_click        = leads       / (clicks + 1)
    conv_per_lead         = conversions / (leads  + 1)
    clicks_per_impression = clicks      / (impressions + 1)

    # Encoded categoricals
    ct_enc  = mappings["Campaign_Type"][campaign_type]
    ta_enc  = mappings["Target_Audience"][target_audience]
    cs_enc  = mappings["Customer_Segment"][customer_segment]

    # Channel binary flags
    channel_flags = {f"channel_{ch}": (1 if ch in selected_channels else 0)
                     for ch in all_channels}

    base = {
        "Conversions_log"       : conversions_log,
        "Leads"                 : leads,
        "Clicks"                : clicks,
        "Impressions"           : impressions,
        "Acquisition_Cost_log"  : acquisition_cost_log,
        "Engagement_Score"      : engagement_score,
        "Total_Cost"            : total_cost,
        "Duration"              : duration,
        "Conv_per_Click"        : conv_per_click,
        "Lead_per_Click"        : lead_per_click,
        "Conv_per_Lead"         : conv_per_lead,
        "Clicks_per_Impression" : clicks_per_impression,
        "Campaign_Type_encoded"     : ct_enc,
        "Target_Audience_encoded"   : ta_enc,
        "Customer_Segment_encoded"  : cs_enc,
        **channel_flags
    }
    return base

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
predict_btn = st.sidebar.button("🚀 Predict", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# Main Panel — Results
# ─────────────────────────────────────────────
if predict_btn:
    raw = build_input()

    # Align features to each model's expected order
    X_reg = pd.DataFrame([[raw[f] for f in reg_feats]], columns=reg_feats)
    X_clf = pd.DataFrame([[raw[f] for f in clf_feats]], columns=clf_feats)

    revenue_pred    = reg_model.predict(X_reg)[0]
    profit_pred     = clf_model.predict(X_clf)[0]
    profit_proba    = clf_model.predict_proba(X_clf)[0]

    # ── Result Cards ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="💰 Predicted Revenue",
            value=f"₹ {revenue_pred:,.0f}"
        )

    with col2:
        label = "✅ Profit" if profit_pred == 1 else "❌ Loss"
        st.metric(label="📈 Campaign Outcome", value=label)

    with col3:
        confidence = profit_proba[profit_pred] * 100
        st.metric(label="🎯 Confidence", value=f"{confidence:.1f}%")

    st.divider()

    # ── Probability Bar ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Outcome"     : ["Loss", "Profit"],
            "Probability" : [profit_proba[0] * 100, profit_proba[1] * 100]
        })
        st.bar_chart(prob_df.set_index("Outcome"))

    with col_b:
        st.subheader("Campaign Summary")
        st.dataframe(pd.DataFrame({
            "Parameter"   : ["Campaign Type", "Target Audience", "Customer Segment",
                             "Channels Used", "Duration", "Clicks",
                             "Leads", "Conversions", "Acquisition Cost", "Total Cost"],
            "Value"       : [campaign_type.title(), target_audience.title(),
                             customer_segment.title(),
                             ", ".join(selected_channels) if selected_channels else "None",
                             f"{duration} days", f"{clicks:,}",
                             f"{leads:,}", f"{conversions:,}",
                             f"₹ {acquisition_cost:,.2f}", f"₹ {total_cost:,.0f}"]
        }), use_container_width=True, hide_index=True)

    # ── ROI Estimate ──
    st.divider()
    if total_cost > 0:
        estimated_roi = (revenue_pred - total_cost) / total_cost
        roi_col1, roi_col2, roi_col3 = st.columns(3)
        roi_col1.metric("Estimated ROI",    f"{estimated_roi:.2f}x")
        roi_col2.metric("Estimated Profit", f"₹ {revenue_pred - total_cost:,.0f}")
        roi_col3.metric("Revenue / Cost",   f"{revenue_pred / (total_cost + 1):.2f}x")

else:
    # Default landing state
    st.info("👈 Fill in the campaign details on the left sidebar and click **Predict** to get results.")

    st.subheader("📌 How to Use")
    st.markdown("""
    1. Select **Campaign Type**, **Target Audience**, and **Customer Segment**
    2. Check the **Channels** used in the campaign
    3. Enter the campaign **numeric metrics** (impressions, clicks, leads, etc.)
    4. Click **🚀 Predict** to get:
        - **Predicted Revenue** from the Regression model
        - **Profit / Loss** outcome from the Classification model
        - **Confidence** score of the prediction
    """)

    st.subheader("🤖 Models Used")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Regression Model** — Random Forest\n\nPredicts Revenue (R² = 0.79)")
    with col2:
        st.success("**Classification Model** — Random Forest\n\nPredicts Profit/Loss (F1 = 94.3%)")