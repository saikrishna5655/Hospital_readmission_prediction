# 🏥 Hospital Readmission Prediction System
> Predicting 30-day hospital readmissions for diabetic patients using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Diabetes%20130--US-green.svg)

---

## 📌 Problem Statement

Hospitals face significant financial penalties when diabetic patients are 
readmitted within 30 days of discharge. This project builds a machine learning 
model to predict which patients are at high risk of readmission, enabling 
doctors to intervene early and provide better care.

**Real-world impact:**
- Hospitals lose up to 3% of Medicare payments for excess readmissions
- Early identification of high-risk patients saves $300-500 per prevented readmission
- Better patient outcomes through timely follow-up care

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| Name | Diabetes 130-US Hospitals (1999-2008) |
| Original Size | 101,766 encounters × 50 columns |
| Unique Patients | 71,518 patients |
| Target Variable | Readmitted within 30 days (Yes/No) |

---

## 🚀 Project Progress

### ✅ Part 1: Data Exploration & Cleaning
- Loaded raw dataset (101,766 rows × 50 columns)
- Discovered `?` symbols masking missing values
- Identified severe class imbalance (only 11% readmitted within 30 days)
- Dropped 5 high-missing columns (weight 96%, glucose 94%, A1C 83%)
- Handled remaining missing values (<3%)
- Final clean dataset: **99,493 rows × 45 columns, 0 missing values**

### ✅ Part 2: Feature Selection & Engineering
- Grouped 45 columns into logical categories (demographics, clinical, medications etc.)
- Dropped 13 near-zero medication columns (<1% usage)
- Simplified remaining medications to binary 0/1 (taking or not taking)
- Dropped ID columns (encounter_id, patient_nbr)
- Converted age ranges [0-10), [10-20)... to midpoint numbers (5, 15, 25...)
- Grouped 682 unique ICD-9 diagnosis codes into 6 disease families
- One-hot encoded categorical columns (race, diagnosis categories)
- Converted all columns to integers
- Final model-ready dataset: **99,493 rows × 45 columns, all integers**

### 🔄 Part 3: Model Training *(Coming Next)*
- Train/test split
- Baseline model
- Handle class imbalance
- Evaluate model performance

### 📝 Part 4: Model Optimization *(Planned)*

### 📝 Part 5: Deployment & Dashboard *(Planned)*

---

## 💡 Key Findings So Far

### 1. Class Imbalance is the Biggest Challenge
Only **11.23%** of patients are readmitted within 30 days.
A model that always predicts "NO" would be 89% accurate but completely useless.
Special techniques needed during model training.

### 2. Patient History is the Strongest Predictor
Prior hospital stays (`number_inpatient`) shows the biggest difference:
| Readmission Status | Avg Prior Inpatient Visits |
|-------------------|--------------------------|
| Readmitted <30 days | 1.23 |
| Readmitted >30 days | 0.84 |
| Not Readmitted | 0.39 |

Patients readmitted within 30 days had **3x more prior hospital stays!**

### 3. Elderly Patients Are Most at Risk
Age group 70-80 has the highest patient count (25,469) and significantly 
higher readmission rates than younger patients.

### 4. Primary Diagnosis Insight
Even in a diabetes dataset, most patients are admitted for Heart conditions 
and "Other" complications - not diabetes directly. Diabetes is often the 
underlying cause, not the immediate reason for admission.

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and cleaning
- **numpy** - Numerical operations
- **matplotlib & seaborn** - Data visualization
- **scikit-learn** - Machine learning *(upcoming)*
- **Google Colab** - Development environment

---

## 📁 Project Structure
hospital-readmission-prediction/
│
├── data/
│ ├── raw/
│ │ └── diabetic_data.csv # Original dataset
│ └── processed/
│ ├── diabetes_data_cleaned.csv # After Part 1 cleaning
│ ├── diabetes_data_featured.csv # After Part 2 feature engineering
│ └── diabetes_ready_for_model.csv # Final model-ready dataset
│
├── notebooks/
│ ├── Hospital_Readmission_Part1_Exploration.ipynb
│ └── Hospital_Readmission_Part2_Feature_Selection.ipynb
│
├── README.md
└── requirements.txt

---

## ⚙️ How to Run

```bash
# 1. Clone the repository

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from UCI repository
# https://archive.ics.uci.edu/dataset/296/

# 4. Run notebooks in order
# Part 1 → Part 2 → Part 3 (coming soon)

📚 Requirements
text
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0


🎓 About This Project
This is a hands-on learning project where I'm building a complete end-to-end
machine learning solution for a real healthcare problem. Each part is
documented step by step to show the thinking process behind every decision -
not just the final code.

What makes this different from typical projects:

Every decision is explained with clinical reasoning

Ethical considerations (removed insurance data to avoid bias)

Focus on business impact, not just model accuracy

Real-world messy data challenges documented honestly
