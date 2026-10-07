# Marketing Campaign Performance Prediction — Project Documentation

This document explains the process followed and the key decisions made at each stage of the project, including why certain approaches were chosen over alternatives.

---

## 1. Business Understanding

**What the data is:** Ad campaign performance data from three competing Indian beauty/cosmetics e-commerce platforms — Nykaa, Purplle, and Tira — each tracked over time.

**Business questions this project answers:**
- How well is each brand's marketing performing (Revenue, ROI, Conversions)?
- Which channels and campaign types work best?
- Can we predict a campaign's expected Revenue before running it?
- Can we predict whether a campaign will be profitable or a loss?

**Key column distinctions clarified early:**
- `Target_Audience` = who the campaign was aimed at. `Customer_Segment` = who actually engaged. These can differ — and in this dataset, **80.1% of campaigns had a mismatch**, a notable finding on its own.
- `Engagement_Score` is a normalized interaction-quality metric, distinct from raw funnel counts (Impressions, Clicks, etc.).

---

## 2. Issue Identification

Before cleaning, the data was audited using a standard framework: structure → missing values → duplicates → categorical consistency → numeric distribution → domain/logic rules.

**Findings:**
- `Date` stored as string, funnel columns (`Impressions`, `Clicks`, `Leads`, `Conversions`) stored as float due to nulls forcing the dtype.
- ~53% of rows had at least one missing value somewhere — too high to drop rows; required column-by-column imputation instead.
- `ROI` was unreliable: comparing stored ROI against the expected formula showed **86% of rows deviated**, confirming the brief's note that ROI needed recalculation.
- A small number of rows broke funnel logic (e.g. `Leads > Clicks`) — genuine raw data errors, later dropped (37 + 341 rows, both negligible against total size).

---

## 3. Data Cleaning — Decisions & Reasoning

| Column(s)                                                                                              | Decision                                                                                                                     | Why                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Campaign_ID`                                                                                          | Dropped after deriving `Brand`                                                                                               | No predictive value once brand identity is captured separately                                                                                                                                                                                                                                                                                                                                                           |
| `Date`                                                                                                 | Converted to datetime, kept in cleaned data, nulls left as-is                                                                | Needed for optional trend analysis in EDA; not required for modeling, so nulls don't block anything                                                                                                                                                                                                                                                                                                                      |
| `Impressions/Clicks/Leads/Conversions`                                                                 | Cast to `Int64` (nullable int)                                                                                               | These are counts — must be whole numbers; nullable int avoids forcing float before imputation                                                                                                                                                                                                                                                                                                                            |
| Categorical nulls (`Campaign_Type`, `Target_Audience`, `Channel_Used`, `Language`, `Customer_Segment`) | Proportional random fill (sample from existing class distribution)                                                           | Avoids artificially inflating one category the way a flat mode-fill would; these columns have weak predictive power for Revenue/Profit anyway, so imputation method here has minimal effect on model accuracy                                                                                                                                                                                                            |
| `Impressions`                                                                                          | Median fill                                                                                                                  | Grouping by Brand/Campaign_Type showed only minor variation — not worth the added complexity                                                                                                                                                                                                                                                                                                                             |
| `Clicks`                                                                                               | Ratio-based fill: `Impressions × CTR` (CTR = median of existing Clicks/Impressions)                                          | Clicks is a function of Impressions, not independent — a flat fill would ignore that relationship and produce illogical rows                                                                                                                                                                                                                                                                                             |
| `Leads`                                                                                                | Ratio-based fill: `Clicks × lead_rate`                                                                                       | Same reasoning — Leads depends on Clicks                                                                                                                                                                                                                                                                                                                                                                                 |
| `Conversions`                                                                                          | Ratio-based fill: `Leads × conversion_rate`                                                                                  | Same reasoning — Conversions depends on Leads                                                                                                                                                                                                                                                                                                                                                                            |
| `Revenue`                                                                                              | Ratio-based fill: `Conversions × revenue_per_conversion`                                                                     | Correlation check showed Revenue is most strongly driven by Conversions (0.88) — strongest available driver column                                                                                                                                                                                                                                                                                                       |
| `Acquisition_Cost`                                                                                     | Median fill                                                                                                                  | Correlation with all numeric columns was weak (max ~-0.48) — this cost is driven by external factors (team/channel/market decisions) not captured in the funnel data, so a ratio-based fill would add false precision                                                                                                                                                                                                    |
| `ROI`                                                                                                  | **Recomputed from scratch** using `(Revenue − Total_Cost) / Total_Cost`, where `Total_Cost = Acquisition_Cost × Conversions` | Original ROI was unreliable (86% error rate vs. formula); recalculating is more correct than trying to impute/patch an untrustworthy column. The need for `Total_Cost` (rather than raw per-unit `Acquisition_Cost`) was discovered after noticing Acquisition_Cost values (hundreds) were far smaller than Revenue values (lakhs) — confirming Acquisition_Cost is a **per-conversion unit cost**, not a campaign total |
| Funnel-logic violations (`Leads > Clicks`, `Conversions > Leads`)                                      | Rows dropped                                                                                                                 | Small number of rows (<1% of data), confirmed as genuine raw data errors on inspection — not worth a more complex fix                                                                                                                                                                                                                                                                                                    |

---

## 4. Exploratory Data Analysis — Key Insights

1. **Brand performance is nearly uniform** — Revenue, ROI, and Conversions are within a tight range across Nykaa, Purplle, and Tira. Brand alone is a weak predictor.
2. **Channel effectiveness is also fairly even** — Instagram and Email edge out slightly on ROI, but the gap is small.
3. **Revenue and ROI are different success dimensions** — the highest-revenue campaign in the dataset had a mediocre ROI (2.28), while a mid-revenue campaign had an ROI of 42.72. High spend buys revenue; efficiency is independent of scale.
4. **Revenue is primarily driven by Conversions** (0.88 correlation), then Leads (0.80) and Clicks (0.71). Acquisition_Cost has a moderate negative relationship with Revenue (−0.41) — efficient (lower-cost) campaigns are not underperforming.
5. **Engagement_Score has a non-linear relationship with Revenue** — the highest-revenue campaigns cluster around a mid-range engagement score (20–30), not the highest scores. This ruled out treating engagement as a simple "more is better" feature.
6. **80.1% of campaigns show a Target_Audience / Customer_Segment mismatch** — a significant targeting-precision finding, independent of the modeling task.

These insights directly shaped feature selection — prioritizing the funnel chain (Impressions → Conversions), Engagement_Score, and cost features as primary predictors.

---

## 5. Feature Engineering — Decisions & Reasoning

- **`Channel_Used` → Multi-label binarization** (one binary column per channel), not one-hot encoding — because a campaign can use multiple channels at once (comma-separated), which one-hot encoding cannot represent correctly.
- **Low-cardinality categoricals** (`Brand`, `Campaign_Type`, `Target_Audience`, `Customer_Segment`, `Language`) → one-hot encoded with `drop_first=True`, since each has only 4–6 unique values — safe from high-dimensional explosion, and dropping one category per column avoids redundant/duplicated information.
- **`Profit_Flag`** created from ROI (`Profit` if ROI > 0, else `Loss`) — found to be imbalanced at roughly 77% Profit / 23% Loss, which was accounted for later by stratifying the train/test split and evaluating the classifier on Loss-class precision/recall specifically, not just overall accuracy.

### Data Leakage — the most important decision in this stage

- `ROI` is excluded from **both** models, since it's mathematically derived from Revenue (`(Revenue − Total_Cost) / Total_Cost`).
- For the **classification** model specifically, `Revenue` and `Total_Cost` are **also excluded**, beyond what the brief explicitly required. Reason: `Profit_Flag` is a direct function of Revenue vs. Total_Cost — giving the model both numbers would let it trivially reconstruct the label rather than learn genuine patterns. This is the same category of leakage as including ROI directly, just less obvious.
- This decision was validated against an external reference: a similar public dataset/study that *did* include ROI as a classification input reported 98% accuracy — almost certainly inflated by leakage. Excluding it here produced a lower but more honest and trustworthy 91.8% accuracy.

---

## 6. Modeling — Revenue Prediction (Regression)

**Approach:** Trained and compared 8 total model configurations — Linear Regression, Random Forest, XGBoost (default), Gradient Boosting, Extra Trees, LightGBM, CatBoost, and a tuned XGBoost (via `RandomizedSearchCV`).

**Result:** Every model converged to approximately the same R² (0.77–0.78), regardless of algorithm complexity or tuning.

**Interpretation:** When a simple Linear Regression (R²=0.77) performs almost identically to a tuned gradient-boosted model, the limiting factor is not model choice — it's an information ceiling in the available features. This traces back directly to the EDA finding that Conversions↔Revenue correlation is 0.88 (which squares to ≈0.77) — the model is essentially recovering that one strong relationship and finding little additional signal elsewhere.

**Decision:** Rather than over-optimizing a ceiling that multiple algorithms confirmed, the tuned XGBoost (R²=0.773, the best of the 8) was accepted as the final model, and this limitation is documented transparently rather than concealed.

---

## 7. Modeling — Profit/Loss Prediction (Classification)

**Approach:** Logistic Regression, Random Forest, and XGBoost were trained and compared, each evaluated on Accuracy plus Precision/Recall/F1 **specifically for the minority "Loss" class** (since the 77/23 class imbalance makes overall accuracy alone a misleading metric — a model that always predicted "Profit" would already score ~77%).

**Result:** All three models landed around 91.6–91.8% accuracy. XGBoost was selected as the best performer (Accuracy = 0.9176, Precision(Loss) = 0.86, Recall(Loss) = 0.76, F1(Loss) = 0.81).

**Decision:** This result meets the project's accuracy target while remaining leakage-free, and was accepted as final given diminishing returns from further tuning.

---

## 8. End-to-End Application (Streamlit)

Built as a multi-page app:
- **Home** — landing page.
- **EDA Dashboard** — KPIs, brand comparison, channel effectiveness, top/bottom campaigns, and a correlation heatmap, reusing the EDA notebook's logic on a lighter, pre-encoding snapshot of the data (so human-readable columns like `Brand` and `Channel_Used` remain available for display).
- **Prediction** — a form collecting campaign details, which are transformed into the exact encoded feature format the models were trained on, before running both the regression and classification models and displaying Predicted Revenue, Profit/Loss outcome with confidence, and an estimated ROI.

**Key implementation decision:** `Total_Cost` in the app is calculated the same way as in training — `Acquisition_Cost × Conversions` — and the saved feature-name lists (`regression_features.pkl` / `classification_features.pkl`) are used to build each model's input row in the exact column order expected, rather than assuming the two models take the same feature set. This directly addresses a mismatch bug identified in an earlier version of this project, where the app used a different formula for `Total_Cost` than training did, causing predictions to be based on inconsistent features.

---

## 9. Summary of Key Decisions

| Decision | Reasoning |
|---|---|
| Ratio-based imputation for the funnel chain | Preserves realistic, row-specific relationships instead of flattening all rows to one value |
| Recompute ROI instead of imputing it | Original column was unreliable (86% formula mismatch); recalculation is more trustworthy than patching |
| Exclude Revenue & Total_Cost (not just ROI) from classification | Prevents indirect leakage of the target label through arithmetic relationship |
| Accept the regression R² ceiling (~0.77) rather than over-fit | Multiple, very different algorithms converged to the same score — a sign of a genuine data/feature limit, not a solvable modeling problem |
| Evaluate classification on Loss-class metrics, not just accuracy | Class imbalance (77/23) makes raw accuracy alone misleading |
