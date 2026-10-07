import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

df = pd.read_csv("data/cleaned/cleaned_data.csv")
st.markdown('<p style="font-size:2.2rem; font-weight:700;">📊 Campaign EDA Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p style="opacity:0.7;">Key insights across brand, channel, and campaign performance</p>', unsafe_allow_html=True)

# ---- KPI row ----
k1, k2, k3 = st.columns(3)
k1.metric("Total Campaigns", f"{len(df):,}")
k2.metric("Avg Revenue", f"₹{df['Revenue'].mean():,.0f}")
k3.metric("Avg ROI", f"{df['ROI'].mean():.2f}x")

st.divider()

# ---- Brand comparison ----
st.subheader("🏢 Brand Performance")
brand_summary = df.groupby('Brand')[['Revenue','ROI','Conversions','Acquisition_Cost']].mean().round(2)
c1, c2 = st.columns([1, 1])
with c1:
    st.dataframe(brand_summary, use_container_width=True)
with c2:
    fig, ax = plt.subplots(figsize=(5, 3))
    brand_summary['Revenue'].plot(kind='bar', ax=ax, color='#6366f1')
    ax.set_ylabel("Avg Revenue")
    st.pyplot(fig)

st.divider()

# ---- Channel effectiveness ----
st.subheader("📣 Channel Effectiveness")
channel_df = df.assign(Channel_Used=df['Channel_Used'].str.split(',')).explode('Channel_Used')
channel_df['Channel_Used'] = channel_df['Channel_Used'].str.strip()
channel_summary = channel_df.groupby('Channel_Used')[['Revenue','ROI','Conversions']].mean().round(2).sort_values('ROI', ascending=False)
st.dataframe(channel_summary, use_container_width=True)

st.divider()

# ---- Top/bottom campaigns ----
st.subheader("🏆 Top & Low Performing Campaigns")
t1, t2 = st.tabs(["Top 10 by Revenue", "Bottom 10 by Revenue"])
with t1:
    st.dataframe(df.nlargest(10, 'Revenue')[['Brand','Campaign_Type','Channel_Used','Revenue','ROI']], use_container_width=True)
with t2:
    st.dataframe(df.nsmallest(10, 'Revenue')[['Brand','Campaign_Type','Channel_Used','Revenue','ROI']], use_container_width=True)

st.divider()

# ---- Correlation heatmap ----
st.subheader("🔗 Spend, Clicks, Revenue & ROI Relationships")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.heatmap(df[['Acquisition_Cost','Clicks','Conversions','Revenue','ROI']].corr(), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)