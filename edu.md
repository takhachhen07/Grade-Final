# GradePric — Student Performance Analytics System
> **Data Warehouse & Data Mining Project**  
> **Team Members:** Rahul, Neeti & Sathin — **022 Batch**

---

## 📌 1. Overview

**GradePric** is an academic analytics web platform built using **Python Flask**, **Pandas**, and **Scikit-Learn**. It applies Data Warehousing and Data Mining techniques to predict student performance (Pass/Fail), clean messy academic data, and provide personalized study recommendations.

---

## 🏬 2. Data Warehouse (DW) Concepts

### A. Operational Data Store (ODS)
- `student_performance.csv` acts as a staging database storing operational student records (Attendance, Study Hours, Internal Marks, Grades, Absences, Result).

### B. ETL (Extract, Transform, Load) Pipeline
1. **Extract**: Ingests raw data from CSV uploads or manual web form submissions.
2. **Transform**:
   - **Imputation**: Missing numerical values are filled with the **Mean**; categorical values with the **Mode**.
   - **Range Enforcement**: Clips numerical values to valid boundaries (e.g., Attendance 0–100%, Internal Marks 0–50).
   - **Encoding**: Converts letter grades and pass/fail labels into numeric formats for model training.
3. **Load**: Saves cleaned datasets to the ODS and feeds them into the machine learning pipeline.

### C. Analytical Aggregations
- Computes key summary stats across the dataset, such as average attendance rate, average study hours, and overall pass/fail percentage.

---

## ⛏️ 3. Data Mining (DM) Concepts

### A. Decision Tree Classification
- Uses a **Decision Tree Classifier** (C4.5 / Entropy) to segment student data into decision rules predicting whether a student will Pass or Fail.

### B. Probability Calibration
- Applies probability calibration (Platt Scaling) to convert raw tree splits into smooth, realistic Pass/Fail probability scores ($0–100\%$).

### C. Feature Importance Analysis
- Measures how much each attribute contributes to the prediction in percentage (%):
  1. **Internal Assessment Marks:** **38.5%**
  2. **Weekly Study Hours:** **28.4%**
  3. **Attendance Rate (%):** **18.2%**
  4. **Previous Letter Grade:** **9.1%**
  5. **Unexcused Absences:** **5.8%**

### D. Rule-Based Academic Interventions
- Uses heuristic rules to generate tailored advice based on risk factors:
  - **Attendance < 75%**: Mandatory academic counseling.
  - **Internal Marks < 20/50**: Remedial tutoring recommended.
  - **Study Hours < 10 hrs/wk**: Study schedule adjustment needed.
  - **Absences > 7 days**: Excessive absenteeism warning.

---

## 💡 4. Key Application Features

1. **Overview Dashboard**: Displays high-level stats (Total Records, Dataset Pass Rate, Model Accuracy, F1-Score).
2. **Student Dataset & ETL Cleaning**: Displays dataset records, flags missing data, enables single-click cleaning, and supports custom CSV uploads.
3. **Outcome Predictor**: Accepts individual student details to compute calibrated Pass/Fail predictions and customized study advice.
4. **Prediction History Log**: Maintains an audit log (`prediction_history.json`) of past predictions.

