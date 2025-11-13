#  Football Match Outcome Prediction — Complete Machine Learning Project

A fully end‑to‑end, production‑ready **Football Match Outcome Prediction System** that uses advanced Machine Learning models like **CatBoost, XGBoost, and Random Forest**. This project predicts the match result — **Win / Draw / Loss** — using rich historical match data and powerful feature engineering.

This README is designed to be **GitHub‑worthy**, clean, structured, and perfect for showcasing your project.

---

# 📌 Project Summary

This project builds a machine learning model that predicts:

* **0 = Loss**
* **1 = Draw**
* **2 = Win**

Using:

* Historical match statistics
* Team performance trends
* Rolling form indicators
* Home/away advantage
* Season and competition data
* Advanced ML models

→ The best performing model was **CatBoostClassifier**, achieving **~73% accuracy**, which is strong for sports predictions.

---

# 📂 Dataset Details

The dataset includes ~3800 matches with columns such as:

* Date
* Team
* Opponent
* Shots
* Shots on target
* Expected Goals (xG)
* Ball possession
* Formation
* Penalties
* Venue (Home/Away)
* Season
* Result (W/D/L)

After cleaning, the final working dataset contains **3102 matches**.

---

# 🧹 Data Cleaning & Preprocessing

### ✔ Dropped irrelevant columns:

* `notes`
* `referee`
* `match report`
* `time`

### ✔ Converted & sorted:

* Converted `date` → datetime
* Sorted by `team + date` to prevent data leakage

### ✔ Removed leakage columns:

* `gf` (goals for)
* `ga` (goals against)
* `goal_diff`

These contain final results → DO NOT USE as features.

### ✔ Null Handling

Initial cleaning removed rows with incomplete stats.

---

# 🛠️ Feature Engineering (The Heart of the Project)

Feature engineering is essential in football analytics.

### 🔵 Rolling Statistics (per team)

These capture a team's **form**:

* `Rolling Average 5 Sh` — avg shots over last 5 matches
* `rolling_avg_sot` — avg shots on target
* `rolling_avg_xg` — avg expected goals
* `Rolling win rate` — percent of recent wins

These features greatly improved predictions.

### 🔵 Categorical Transformation

Converted match text info into numerical format:

* Competition
* Round
* Day
* Opponent
* Team
* Season

### 🔵 Home Advantage

```
is_home = 1 if Home else 0
```

Home advantage is statistically significant.

### 🔵 Final Feature Set

* Categorical (encoded)
* Numeric (scaled)
* Rolling stats
* Match context features

---

# 🔢 Encoding & Scaling

### Label Encoding

Used for categorical features:

* comp, round, day, opponent, team, season

### StandardScaler

Used for numerical features:

* xg, xga, sh, sot, dist, fk, pk, pkatt
* All rolling stats

CatBoost does not require encoding but dataset consistency is maintained.

---

# 🤖 Machine Learning Models Used

### Models trained:

* **RandomForestClassifier**
* **XGBClassifier**
* **CatBoostClassifier** (BEST)
* Logistic Regression
* KNN

### Hyperparameter Tuning

Used:

* `GridSearchCV`
* `RandomizedSearchCV`

Best tuned parameters were obtained for XGBoost & CatBoost.

---

# 🏆 Best Model: CatBoostClassifier

CatBoost performed best due to:

* Superior handling of categorical features
* Ordered boosting → less overfitting
* Works well with tabular, mixed-type features
* Handles missing values internally
* Powerful for sports analytics

### 📈 Final Accuracy: **~73%**

Consistent across validation and test sets.

Draw prediction is naturally lower (common issue in football ML projects).

---

# 📊 Model Evaluation

### Metrics Used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Confusion Matrix Performance:

* Very strong for predicting **Win** and **Loss**
* Draw remains hardest (expected)

---

# 🚀 Deployment Preparation

To deploy the project, save:

### 🔹 Model

```
cat_model.save_model("model.cbm")
```

### 🔹 Encoders

```
joblib.dump(encoders, "encoders.pkl")
```

### 🔹 Scaler

```
joblib.dump(scaler, "scaler.pkl")
```

### 🔹 Final Features List

Required for prediction pipelines.

---

# 📁 Project Folder Structure (Recommended)

```
project/
│
├── data/
│ └── final_matches.csv
│
├── notebooks/
│ ├── 1_raw_exploration.ipynb
│ ├── 2_model_training.ipynb
│ └── 3_visualization.ipynb
│
models/
│ ├── catboost_model.cbm
│ ├── rf_model.pkl
│ ├── xgb_best.pkl
│ ├── scaler.pkl
│ ├── encoders.pkl
│ └── features.json
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 📦 requirements.txt

```
pandas
numpy
scikit-learn
xgboost
catboost
matplotlib
seaborn
joblib
```

---

# 🧠 Key Learnings

* Prevent data leakage using proper sorting
* Use rolling windows for team form
* Encode categorical info carefully
* Scale numerical values
* CatBoost often wins tabular ML problems
* Draw classes are always hardest
* Hyperparameter tuning improves stability
* Save preprocessing artifacts for deployment

---

# ✨ Author

**Vinayak Tanwar**
Machine Learning & Data Science Enthusiast

If you like this project, ⭐ star the repository on GitHub!
