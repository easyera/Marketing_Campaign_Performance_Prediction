# 📊 Marketing Campaign Performance Prediction

An end-to-end Machine Learning project to analyze and predict marketing campaign performance across multiple brands (Nykaa, Purplle, Tira).

---

## 🎯 Project Overview

Marketing teams generate large volumes of campaign data — impressions, clicks, conversions, revenue, and ROI. This project transforms that raw data into a structured ML pipeline that:

- Predicts **Revenue** using a Regression model
- Predicts **Profit / Loss** outcome using a Classification model
- Deploys predictions through an interactive **Streamlit dashboard**

---

## 🗂️ Project Structure

```
Marketing Campaign Performance Prediction/
│
├── data/
│   ├── raw/                        # Original raw CSV files
│   ├── cleaned/                    # After cleaning & deduplication
│   └── processed/                  # Model-ready dataset
│
├── notebooks/
│   ├── data_preprocessing.ipynb    # Cleaning, feature engineering, encoding
│   ├── eda.ipynb                   # Exploratory Data Analysis
│   ├── regression_model.ipynb      # Revenue prediction model
│   └── classification_model.ipynb  # Profit/Loss prediction model
│
├── model/
│   └── parameters/
│       ├── regression_model.pkl        # Trained regression model
│       ├── classification_model.pkl    # Trained classification model
│       ├── regression_features.pkl     # Feature list for regression
│       ├── classification_features.pkl # Feature list for classification
│       └── label_mappings.json         # Encoding maps for categorical inputs
│
├── main.py                         # Streamlit dashboard app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/marketing-campaign-prediction.git
cd marketing-campaign-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate model files

Since `.pkl` model files are excluded from Git (large binaries), you need to generate them locally:

```bash
# Step 1 — Run preprocessing notebook to generate the processed dataset
jupyter notebook notebooks/data_preprocessing.ipynb

# Step 2 — Run regression notebook to train and save regression_model.pkl
jupyter notebook notebooks/regression_model.ipynb

# Step 3 — Run classification notebook to train and save classification_model.pkl
jupyter notebook notebooks/classification_model.ipynb
```

After running all three notebooks, your `model/parameters/` folder will have all required files.

---

## 🚀 Running the Streamlit App

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🖥️ How to Use the Dashboard

1. **Fill in campaign details** in the left sidebar:
   - Select **Campaign Type** (Email, Influencer, Paid Ads, SEO, Social Media)
   - Select **Target Audience** and **Customer Segment**
   - Check the **Channels Used** (Facebook, Google, Instagram, etc.)
   - Enter numeric inputs: Duration, Impressions, Clicks, Leads, Conversions, Acquisition Cost, Engagement Score

2. **Click 🚀 Predict**

3. **View Results:**
   - 💰 **Predicted Revenue** — estimated revenue from the campaign
   - 📈 **Campaign Outcome** — Profit ✅ or Loss ❌
   - 🎯 **Confidence** — how confident the model is in its prediction
   - **Probability chart** — visual breakdown of Profit vs Loss probability
   - **Estimated ROI** — auto-calculated from predicted revenue and cost

---

## 🤖 Models

| Task | Model | Metric | Score |
|---|---|---|---|
| Revenue Prediction | Random Forest Regressor | R² | 0.79 |
| Profit/Loss Classification | Random Forest Classifier | F1-Score | 0.94 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | Model training |
| Matplotlib & Seaborn | EDA visualizations |
| Plotly | Interactive charts |
| Streamlit | Dashboard deployment |
| Jupyter Notebook | Development environment |

---

## 📦 Dataset

The dataset contains marketing campaign data with the following key columns:

`Campaign_Type`, `Target_Audience`, `Channel_Used`, `Impressions`, `Clicks`, `Leads`, `Conversions`, `Revenue`, `Acquisition_Cost`, `ROI`, `Engagement_Score`, `Customer_Segment`, `Duration`

> Dataset source: [Marketing Campaign Performance Prediction Datasets](https://drive.google.com/drive/folders/1hZBFlErfcTQ5G8o9TmrMk1ZJl5ZpeEkV)

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)