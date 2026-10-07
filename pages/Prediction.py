import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

st.set_page_config(page_title="Campaign Predictor", page_icon="📈", layout="wide")

# load regression artifacts
with open('model/regression_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)
with open('model/regression_features.pkl', 'rb') as f:
    reg_features = pickle.load(f)

# load classification artifacts
with open('model/classification_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)
with open('model/classification_features.pkl', 'rb') as f:
    clf_features = pickle.load(f)
with open('model/label_mappings.json', 'r') as f:
    label_map = json.load(f)

# ---- Custom styling ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stForm"] {
        background-color: rgba(128, 128, 128, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 Campaign Performance Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter campaign details to predict Revenue and Profit/Loss outcome</p>', unsafe_allow_html=True)

# ---- Form (groups inputs + adds a clean submit button) ----
with st.form("prediction_form"):
    st.subheader("🏷️ Campaign Profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        brand = st.selectbox("Brand", ["Nykaa", "Purplle", "Tira"])
        campaign_type = st.selectbox("Campaign Type", ["Influencer", "Paid Ads", "Seo", "Social Media", "Email"])
    with c2:
        target_audience = st.selectbox("Target Audience", ["Premium Shoppers", "Tier 2 City Customers", "Working Women", "Youth", "College Students"])
        customer_segment = st.selectbox("Customer Segment", ["Premium Shoppers", "Tier 2 City Customers", "Working Women", "Youth", "College Students"])
    with c3:
        language = st.selectbox("Language", ["English", "Hindi", "Tamil"])
        duration = st.number_input("Duration (days)", min_value=1, max_value=60, value=18)

    st.divider()
    st.subheader("📊 Funnel Metrics")
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        impressions = st.number_input("Impressions", min_value=0, value=50000)
    with c5:
        clicks = st.number_input("Clicks", min_value=0, value=4000)
    with c6:
        leads = st.number_input("Leads", min_value=0, value=800)
    with c7:
        conversions = st.number_input("Conversions", min_value=0, value=200)

    st.divider()
    st.subheader("💰 Cost & Engagement")
    c8, c9 = st.columns(2)
    with c8:
        acquisition_cost = st.number_input("Acquisition Cost (per conversion)", min_value=0.0, value=350.0)
    with c9:
        engagement_score = st.number_input("Engagement Score", min_value=0.0, max_value=100.0, value=25.0)

    st.divider()
    st.subheader("📣 Channels Used")
    channel_options = ["Email", "Facebook", "Google", "Instagram", "Whatsapp", "Youtube"]
    selected_channels = st.multiselect("Select all channels used", channel_options)

    submitted = st.form_submit_button("🔮 Predict Performance", use_container_width=True)

if submitted:
    total_cost = acquisition_cost * conversions

    input_dict = {
        'Duration': duration,
        'Impressions': impressions,
        'Clicks': clicks,
        'Leads': leads,
        'Conversions': conversions,
        'Acquisition_Cost': acquisition_cost,
        'Engagement_Score': engagement_score,
        'Total_Cost': total_cost,
    }

    for ch in ["Email", "Facebook", "Google", "Instagram", "Whatsapp", "Youtube"]:
        input_dict[ch] = 1 if ch in selected_channels else 0

    for b in ["Purplle", "Tira"]:
        input_dict[f"Brand_{b}"] = 1 if brand == b else 0

    for ct in ["Influencer", "Paid Ads", "Seo", "Social Media"]:
        input_dict[f"Campaign_Type_{ct}"] = 1 if campaign_type == ct else 0

    for ta in ["Premium Shoppers", "Tier 2 City Customers", "Working Women", "Youth"]:
        input_dict[f"Target_Audience_{ta}"] = 1 if target_audience == ta else 0

    for cs in ["Premium Shoppers", "Tier 2 City Customers", "Working Women", "Youth"]:
        input_dict[f"Customer_Segment_{cs}"] = 1 if customer_segment == cs else 0

    for lang in ["English", "Hindi", "Tamil"]:
        input_dict[f"Language_{lang}"] = 1 if language == lang else 0

    # Build the row in the EXACT column order the model expects
    X_reg_input = pd.DataFrame([[input_dict[col] for col in reg_features]], columns=reg_features)
    X_clf_input = pd.DataFrame([[input_dict[col] for col in clf_features]], columns=clf_features)

    # ---- Predict ----
    predicted_revenue = reg_model.predict(X_reg_input)[0]
    predicted_class = clf_model.predict(X_clf_input)[0]
    predicted_proba = clf_model.predict_proba(X_clf_input)[0]

    # label_map was {'Loss': 0, 'Profit': 1} — reverse it to decode the prediction
    inv_label_map = {v: k for k, v in label_map.items()}
    predicted_label = inv_label_map[predicted_class]
    confidence = predicted_proba[predicted_class] * 100

    # ---- Display results ----
    st.divider()
    st.subheader("🎯 Prediction Results")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Predicted Revenue", f"₹{predicted_revenue:,.0f}")

    with r2:
        if predicted_label == "Profit":
            st.success(f"✅ {predicted_label}")
        else:
            st.error(f"⚠️ {predicted_label}")

    with r3:
        st.metric("Confidence", f"{confidence:.1f}%")

    predicted_roi = (predicted_revenue - total_cost) / total_cost
    st.metric("Estimated ROI", f"{predicted_roi:.2f}x")