# Heart Attack Prediction using Machine Learning  
Supervised Learning Project — University Assignment

This project develops a machine learning system that predicts the likelihood of heart disease based on clinical patient data.  
It includes:

- Jupyter Notebook (EDA → Preprocessing → Feature Engineering → Model Training → Evaluation)
- A deployed Streamlit web application for interactive risk prediction
- Reproducible environment (`requirements.txt`)
- Version control with meaningful commits

---

## Project Overview

Cardiovascular disease remains one of the leading causes of death globally. Early detection enables faster treatment and prevention.  
This project uses supervised machine learning algorithms to build a **binary classifier** that predicts:

- **0 → No Heart Disease**
- **1 → Heart Disease Present (any severity)**

We convert a multi-class dataset into binary because early diagnosis focuses on distinguishing *presence* vs *absence* of disease.

---

## Repository Structure
ML_Project/
│
├── app.py # Streamlit web application
├── model_pipeline.pkl # Serialized ML pipeline (preprocessing + model)
├── Heart_Attack.csv # Dataset used in the notebook
├── requirements.txt # Reproducible environment
├── ML_Project.ipynb # Main analysis notebook
└── README.md # Project documentation

---

## 📊 Dataset

- **Source:** Mendeley Data  
- **Link:** https://data.mendeley.com/datasets/yrwd336rkz/2  
- **Rows:** 1763 patients  
- **Features:** 14 clinical indicators  
- **Target:** Heart disease presence (initially 0,1,2 → merged to binary)

### ✔ Why Binary Classification?
Classes `1` and `2` both indicate disease presence.  
Merging them reduces noise and improves medical interpretability.

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook includes:

### **Data Shape, Types, Summary**
- Dataset dimensions  
- Distribution of each feature  
- Outlier inspection  
- Class imbalance analysis

### **Visualizations**
- Histograms of key medical variables  
- Boxplots grouped by disease status  
- Correlation heatmaps  
- Scatter plots for important relationships  
- Target distribution plots  

These help identify patterns such as:
- lower peak heart rate in disease patients  
- higher cholesterol/blood pressure levels  
- age and oldpeak correlations

---

## 🛠 Data Preparation

### **Cleaning**
- Replaced *medically impossible values* (e.g., 0 cholesterol) with the median.
- Removed duplicate rows.

### **Feature Engineering**
Created 5 medically-meaningful features:

| Feature | Description |
|--------|-------------|
| `age_group` | Age → 4 risk bins |
| `age_chol_interaction` | Combined aging & cholesterol effect |
| `heart_rate_reserve` | HR difference from expected maximum |
| `bp_category` | Blood pressure risk class |
| `chol_risk` | Cholesterol risk class |

### **Splitting**
- 70% training  
- 15% validation  
- 15% test  

### **Scaling**
Applied **StandardScaler** to numerical features.

---

## Machine Learning Models

Three supervised learning models were trained and compared:

1. **Logistic Regression**  
   - Best performing model  
   - High interpretability  
   - Hyperparameter tuning: C, solver

2. **Random Forest Classifier**  
   - Stable, robust  
   - Provides feature importance

3. **XGBoost Classifier**  
   - Strong performance on tabular data  
   - Tuned using grid search (depth, lr, n_estimators)

---

## Model Evaluation

Metrics used:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**
- Confusion Matrices
- ROC Curves

### Final Results (Test Set)

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.7864 | 0.8120 | **0.8654** |
| XGBoost | 0.7476 | 0.7778 | 0.8576 |
| Random Forest | 0.7039 | 0.7382 | 0.8040 |

 **Logistic Regression achieved the best overall performance**  
→ Selected as the final deployed model.

---

## Deployment (Streamlit App)

The final system is deployed as a **Streamlit web app** with three pages:

### Home  
Project introduction.

### Data Overview  
Displays dataset preview & summary statistics.

### Predict  
Users enter clinical values (age, blood pressure, cholesterol, etc.)  
→ Model outputs **heart disease risk prediction**.

### Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py

