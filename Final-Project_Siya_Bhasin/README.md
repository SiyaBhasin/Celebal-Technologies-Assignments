# Lead Scoring Model — X Education
### Celebal Technologies Data Science Internship | Week 9 Final Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset)

---

## Problem Statement

X Education sells online courses to industry professionals but struggles with a **~30% lead conversion rate**. The sales team wastes time calling every lead equally, regardless of their likelihood to convert.

The goal is to build a **data-driven Lead Scoring Model** that assigns each prospect a score from **0 to 100** based on their conversion likelihood — enabling the sales team to prioritise high-potential leads and push the conversion rate toward the **CEO's target of 80%**.

**Dataset:** [Kaggle — Lead Scoring Dataset](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset) | 9,240 records | 37 features | Target: `Converted` (0/1)

---

## Methodology

### 1. Exploratory Data Analysis
- Confirmed ~30% baseline conversion rate and class imbalance
- Identified and dropped columns with >40% missing values
- Analysed conversion rates across Lead Source, Last Activity, and Occupation
- Plotted numerical distributions split by conversion status

### 2. Data Preprocessing
- Replaced form default `'Select'` values with `NaN`
- Imputed categoricals with **mode**, numericals with **median**
- Binary Yes/No columns mapped to 1/0
- Top-10 categories kept per column, rest grouped as `'Other'`
- One-hot encoding applied, low-variance columns removed

### 3. Feature Selection
- Trained a Random Forest to extract feature importances
- Retained only features with importance **> 0.01** (~30 features)

### 4. Model Training
Three models trained with `class_weight='balanced'` to handle class imbalance:

| Model | Why Used |
|---|---|
| **Logistic Regression** | Interpretable coefficients, clean probability output |
| **Random Forest** | Handles non-linearity, robust to outliers |
| **Gradient Boosting** | Highest accuracy, corrects errors sequentially |

**Final Score** = Average of all three model probabilities × 100

### 5. Threshold Tuning
Swept thresholds from 0.10 → 0.95 to find the optimal cutoff where **precision ≈ 80%** — meaning when the model flags a lead, it's right 80% of the time.

---

## Results

| Model | Accuracy | ROC-AUC | F1-Score |
|---|---|---|---|
| Logistic Regression | ~88% | ~0.95 | ~0.85 |
| Random Forest | ~90% | ~0.96 | ~0.87 |
| Gradient Boosting | ~90% | ~0.96 | ~0.87 |
| **Ensemble (Final)** | **~90%** | **0.961** | **~0.87** |

**At optimal threshold (0.32):**
- Precision (conversion rate on called leads): **~80%**
- Leads to contact: **4,152 of 9,240** (45% of pipeline)
- Wasted calls avoided: **6,366**

---

## Lead Scoring System

| Score | Category | Sales Action |
|---|---|---|
| 75–100 | 🟢 Hot Lead | Contact immediately — highest priority |
| 50–74 | 🟠 Warm Lead | Follow up within 24 hours |
| 25–49 | 🔵 Cold Lead | Add to nurture email campaign |
| 0–24 | 🔴 Very Cold Lead | Remove from active sales queue |

---

## Key Business Insights

**Top factors that INCREASE conversion probability:**
- `Last Activity = SMS Sent` — responded to outreach → high intent signal
- `Occupation = Working Professional` — can afford and needs the course
- `Total Time Spent on Website` — more browsing = genuine interest
- `Tags = Will revert after reading email` — engaged with content

**Top factors that DECREASE conversion probability:**
- `Tags = Ringing` — phone unanswered → unreachable or uninterested
- `Tags = Interested in other courses` — shopping elsewhere
- `Lead Source = Olark Chat with 0 time on site` — no real engagement

**The Precision-Recall Trade-off:**
Targeting 80% precision means missing a few edge-case converters (lower recall) — but this is the right business decision. Every saved call = time redirected to a genuinely hot lead. Revenue per sales-hour goes up significantly.

---

## Streamlit CRM App

The model is deployed as an interactive CRM dashboard with 3 views:

**Executive Dashboard** — KPI cards (total leads, hot leads, avg score, AUC), Before vs After business impact, lead category chart, and top conversion drivers table.

**Sales Lead List** — Full scored dataset sorted by lead score with filters (score slider, category, lead source). Color-coded rows. Download filtered leads as CSV.

**Live Lead Predictor** — Input a new prospect's behaviour via dropdowns and sliders → instant score (0–100), category badge, and specific sales action recommendation.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open Lead_Scoring_Notebook.ipynb → Run All cells
#    (generates lead_scoring_model.pkl and leads_scored.csv)

# 3. Launch the app
python -m streamlit run app.py
# Opens at http://localhost:8501
```

---

## Project Structure

```
Final-Project_Siya_Bhasin/
├── Lead_Scoring_Notebook.ipynb   # Complete ML pipeline (LMS submission)
├── app.py                        # Streamlit CRM dashboard
├── lead_scoring_dataset.csv      # Raw dataset (9,240 leads)
├── leads_scored.csv              # All leads scored and categorised
├── lead_scoring_model.pkl        # Saved model bundle (LR + RF + GB + scaler)
└── requirements.txt              # Python dependencies
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.x** | Core programming language |
| **Pandas** | Data manipulation and preprocessing |
| **NumPy** | Numerical operations |
| **Scikit-learn** | ML models, feature selection, metrics |
| **Streamlit** | Interactive CRM web application |
| **Joblib** | Model serialization (save/load `.pkl`) |
| **Matplotlib / Seaborn** | EDA visualizations in notebook |
| **Kaggle Notebooks** | Development and training environment |

---

## Author
**Siya Bhasin** | B.Tech CSE | Amity University, Noida
Data Science Intern — Celebal Technologies | June–July 2026

*Celebal Technologies Data Science Internship | Final Project | June–July 2026*

