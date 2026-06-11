# 🏥 Hospital Readmission Risk Prediction System

> Predicting 30-day hospital readmissions for diabetic patients using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Diabetes%20130--US-green.svg)
![Recall](https://img.shields.io/badge/Recall-89%25-blue.svg)
![ROC AUC](https://img.shields.io/badge/ROC--AUC-0.658-yellow.svg)

---

## 📌 Problem Statement

Hospitals face significant financial penalties when diabetic patients are readmitted within 30 days of discharge. This project builds a machine learning model to predict which patients are at high risk of readmission, enabling doctors to intervene early and provide better care.

**Real-world impact:**
- Hospitals lose up to 3% of Medicare payments for excess readmissions
- Early identification of high-risk patients saves $300–500 per prevented readmission
- Better patient outcomes through timely follow-up care

---

## 📊 Dataset

- **Source:** UCI Diabetes 130-US Hospitals Dataset
- **Size:** ~100,000 patient encounter records
- **Target Variable:** Readmitted within 30 days (Yes / No)
- **Class Imbalance:** ~88% Not Readmitted vs ~12% Readmitted

**Key Features Used:**

| Feature | Description |
|---|---|
| `num_lab_procedures` | Number of lab tests performed |
| `num_medications` | Number of medications prescribed |
| `time_in_hospital` | Length of hospital stay (days) |
| `age` | Age group of the patient |
| `number_inpatient` | Number of previous inpatient visits |
| `discharge_disposition_id` | Where the patient was discharged to |
| `number_diagnoses` | Total diagnoses recorded |
| `admission_type_id` | Type of admission |

---

## 🧠 ML Pipeline

```
Raw Data
   ↓
Data Cleaning & Missing Value Handling
   ↓
Feature Engineering & Encoding
   ↓
Train / Test Split (80/20)
   ↓
Class Imbalance Handling (SMOTE + scale_pos_weight)
   ↓
Model Training (XGBoost)
   ↓
Threshold Tuning
   ↓
Model Export (.pkl)
   ↓
Ready for Deployment
```

---

## ⚙️ Model Experiments

| Model | Accuracy | Recall (High Risk) | ROC-AUC | Notes |
|---|---|---|---|---|
| Random Forest (default) | 88.74% | 0.01 | 0.639 | Useless for minority class |
| Random Forest (balanced) | 88.74% | 0.00 | 0.639 | No improvement |
| Random Forest + SMOTE | 87% | 0.04 | 0.639 | Minimal gain |
| **XGBoost + scale_pos_weight** | **68%** | **0.51** | **0.658** | **Best model** |
| XGBoost (threshold = 0.3) | — | **0.89** | 0.658 | Final deployed model |

---

## 🎯 Final Model Performance

**Model:** XGBoost Classifier  
**Threshold:** 0.3  
**scale_pos_weight:** 7.91

```
Classification Report (Threshold = 0.3)

                  precision    recall    f1-score
Not Readmitted      0.96        0.70       0.81
Readmitted <30      0.13        0.89       0.23

ROC-AUC Score: 0.658
```

**Patient Impact Comparison:**

| Metric | Old Model | Final Model |
|---|---|---|
| High-risk patients caught | 22 / 2,234 | 1,988 / 2,234 |
| Patients missed | 2,212 | 246 |
| False alarms | 43 | 13,304 |

> At threshold 0.3, the model catches 89% of all high-risk patients — 1,966 more patients flagged for timely intervention compared to the baseline.

---

## 📈 Feature Importance

Top 10 most influential features from the trained model:

```
num_lab_procedures     ████████████  0.1198
num_medications        ██████████    0.1025
time_in_hospital       ███████       0.0732
age                    ██████        0.0611
number_inpatient       █████         0.0573
discharge_disp_id      █████         0.0513
num_procedures         █████         0.0508
number_diagnoses       ████          0.0487
admission_type_id      ███           0.0354
admission_source_id    ███           0.0300
```

---

## 🚦 Threshold Guide for Hospitals

| Threshold | Patients Caught | Patients Missed | False Alarms | Best For |
|---|---|---|---|---|
| 0.5 | 1,138 | 1,096 | 5,184 | Small teams, low capacity |
| 0.4 | 1,647 | 587 | 9,333 | Medium-sized teams |
| **0.3** | **1,988** | **246** | **13,304** | **Recommended default** |
| 0.15 | 2,204 | 30 | 16,162 | Zero-miss critical care |

---

## 📁 Project Structure

```
hospital-readmission-prediction/
│
├── data/
│   └── diabetic_data.csv
│
├── notebooks/
│   └── readmission_model.ipynb
│
├── output/
│   └── readmission_model.pkl
│
├── app/
│   └── app.py
│
├── README.md
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | Preprocessing, metrics, Random Forest |
| XGBoost | Final classifier |
| Imbalanced-learn | SMOTE oversampling |
| Matplotlib & Seaborn | Visualizations |
| Pickle | Model serialization |
| Streamlit | Web app (upcoming) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hospital-readmission-prediction.git
cd hospital-readmission-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook notebooks/readmission_model.ipynb
```

### 4. Run inference on new data

```python
import pickle

with open('output/readmission_model.pkl', 'rb') as f:
    model_package = pickle.load(f)

model     = model_package['model']
threshold = model_package['threshold']

probability = model.predict_proba(X_new)[:, 1]
prediction  = (probability >= threshold).astype(int)

# 1 = High Risk, 0 = Low Risk
```

---

## 🔮 Future Improvements

- [ ] Build Streamlit web app for live predictions
- [ ] Add SHAP values for patient-level explainability
- [ ] Try LightGBM and CatBoost
- [ ] Add temporal features from patient history
- [ ] Deploy on Streamlit Cloud or Hugging Face Spaces
- [ ] Add model monitoring for data drift

---

## 📚 Key Learnings

- **Accuracy is misleading** on imbalanced datasets — 88.74% accuracy while being completely useless for the minority class
- **Recall beats precision** in healthcare — missing a high-risk patient costs far more than a false alarm
- **ROC-AUC reflects true model skill** — the baseline model had a ceiling of ~0.66 due to feature quality, not the algorithm
- **Threshold tuning is a business decision** — the right threshold depends on hospital team capacity, not just math

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🙌 Acknowledgments

- Dataset: [UCI Machine Learning Repository — Diabetes 130-US Hospitals](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
- Inspired by real-world hospital readmission reduction programmes under the CMS Hospital Readmissions Reduction Program (HRRP)

---

*Built as part of a real-time business ML project — from raw data to deployed model.*
