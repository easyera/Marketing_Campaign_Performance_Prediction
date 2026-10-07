## 🗂️ Project Structure

Marketing Campaign Performance Prediction/
│
├── data/
│   ├── raw/                         # Original raw CSVs (Nykaa, Purplle, Tira)
│   ├── cleaned/                     # Cleaned, funnel-validated data
│   └── processed/                   # Processed data (with missing values)
│
├── notebooks/
│   ├── Initial_analysis.ipynb       # Business understanding, column exploration
│   ├── Cleaning.ipynb          # Imputation, funnel validation, ROI 
│   ├── EDA.ipynb                    # Exploratory analysis, 10 documented insights
│   ├── Feature_engineering.ipynb    # Multi-label + one-hot encoding, Profit_Flag
│   ├── Regression_model.ipynb       # Revenue prediction model
│   └── Classification_model.ipynb   # Profit/Loss prediction model
│
├── model/
│   ├── regression_model.pkl
│   ├── classification_model.pkl
│   ├── regression_features.pkl
│   ├── classification_features.pkl
│   └── label_mappings.json
│
├── app.py                          # Streamlit entry point
├── pages/
│   ├── EDA_Dashboard.py
│   └── Prediction.py
├── requirements.txt
├── .gitignore
└── README.md


## 🚀 Running the Streamlit App

```bash
streamlit run app.py
```

## 🖥️ How to Use the Dashboard

**EDA Dashboard page:** View KPIs (total campaigns, avg revenue, avg ROI), brand comparison, channel effectiveness, top/bottom campaigns, and spend-revenue-ROI correlations.

**Prediction page:**
1. Fill in campaign details — Brand, Campaign Type, Target Audience, Customer Segment, Language, Duration, funnel metrics (Impressions through Conversions), Acquisition Cost, Engagement Score, and Channels Used
2. Click **🔮 Predict Performance**
3. View results: Predicted Revenue, Profit/Loss outcome, Confidence %, and Estimated ROI

| Task                       | Model           | Metric   | Score |
|  ---                       |  ---            |   ---    |  ---  |
| Revenue Prediction         | XGBoost (tuned) | R²       | 0.77  |
| Profit/Loss Classification | XGBoost         | Accuracy | 0.92  |

## 📌 Notes

- Revenue regression R² plateaus around 0.77 across 8 tested algorithms (Linear, Random Forest, XGBoost, LightGBM, CatBoost, tuned variants) — indicates a feature/data ceiling rather than a model-selection issue.
- Classification excludes ROI and Total_Cost as features to avoid data leakage, since Profit/Loss is directly derived from ROI.

