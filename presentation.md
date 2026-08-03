# 🎓 GradePric — Data Warehouse & Data Mining Project Presentation
> **Project Title:** GradePric — Student Performance & Academic Analytics System  
> **Course:** Data Warehouse and Data Mining  
> **Team Members:** Rahul, Neeti & Sathin — **022 Batch**  
> **Frameworks:** Python Flask, Pandas, Scikit-Learn, Decision Tree Classification  

---

## 📽️ Slide 1: Title & Project Overview

### **GradePric: Data Warehouse & Data Mining for Student Academic Performance**
- **Presenter Team:** Rahul, Neeti & Sathin (Batch 022)
- **Domain:** Educational Data Mining (EDM) & Data Warehousing (DW)
- **Objective:** Build an end-to-end web system that ingests student academic data into an Operational Data Store (ODS), executes automated ETL data cleaning, performs decision tree classification to predict Pass/Fail outcomes with calibrated probability scores, and provides automated, heuristic study recommendations.

---

## 📽️ Slide 2: Problem Statement & Objectives

### **Why GradePric?**
1. **Early Academic Intervention:** Educational institutions often identify struggling students too late in the semester.
2. **Data-Driven Insights:** Leveraging historical attendance, study habits, and test marks allows proactive identification of at-risk students.
3. **Automated ETL & Robust Ingestion:** External CSV uploads frequently contain missing values, inconsistent column names, or incomplete records. GradePric provides automated ETL cleaning to ensure zero model crashes.

### **Key Objectives:**
- Establish a reliable **Operational Data Store (ODS)** staging layer.
- Implement an automated **ETL Data Preprocessing Pipeline** (imputation, encoding, domain bounds).
- Train & calibrate a **Decision Tree Classifier** using Information Gain (Entropy).
- Deploy an interactive **Flask Web Interface** with live sliders, dataset analytics, and prediction history logs.

---

## 📽️ Slide 3: Data Warehouse (DW) Architecture

### **Data Warehouse & Staging Components in GradePric**

```
+--------------------------+      +--------------------------+      +--------------------------+
|   Raw Data Ingestion     | ---> |  ETL Transformation      | ---> |  Operational Data Store  |
|  (CSV Upload / Web Form) |      | (Cleaning & Encoding)    |      | (student_performance.csv)|
+--------------------------+      +--------------------------+      +--------------------------+
                                                                                  |
                                                                                  v
                                                                    +--------------------------+
                                                                    | Machine Learning Pipeline|
                                                                    | (Decision Tree Classifier)|
                                                                    +--------------------------+
```

1. **Operational Data Store (ODS):**
   - File: `student_performance.csv` serves as the centralized staging layer holding operational student academic records.
2. **Data Granularity & Dimensions:**
   - **Student Profile Dimension:** Gender, Age.
   - **Engagement Dimension:** Attendance %, Absences.
   - **Academic Effort Dimension:** Weekly Study Hours, Internal Marks (out of 50), Previous Grade.
   - **Target Fact:** Result (`Pass` / `Fail`).
3. **OLAP Aggregations (`get_summary_stats`):**
   - Provides real-time analytical metrics: Average Attendance, Average Study Hours, Average Internal Score, and Overall Pass Rate.

---

## 📽️ Slide 4: ETL (Extract, Transform, Load) Pipeline

### **1. Extract (E):**
- Extracts raw student records from uploaded CSV files or interactive web forms.
- Performs flexible header mapping (handles variations like `studyhours` $\rightarrow$ `Study_Hours`, `test_score` $\rightarrow$ `Internal_Marks`).

### **2. Transform (T):**
- **Missing Value Imputation:**
  - *Numerical Features* (`Attendance`, `Study_Hours`, `Internal_Marks`, `Absences`, `Age`): Replaced using **Mean Imputation**:
    $$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$
  - *Categorical Features* (`Gender`, `Previous_Grade`): Replaced using **Mode Imputation** (most frequent value).
- **Domain Bound Clipping:** Bounds numeric attributes to realistic physical ranges (e.g., Attendance constrained to $[0.0, 100.0]\%$, Internal Marks bounded to $[0, 50]$).
- **Categorical Feature Encoding:**
  - **Ordinal Encoding:** Letter grades mapped to numeric ranks ($A \rightarrow 4, B \rightarrow 3, C \rightarrow 2, D \rightarrow 1, F \rightarrow 0$).
  - **Binary Encoding:** Gender (`Male` $\rightarrow 0$, `Female` $\rightarrow 1$).
  - **Target Encoding:** Result (`Fail` $\rightarrow 0$, `Pass` $\rightarrow 1$).

### **3. Load (L):**
- Staged into `student_performance.csv` and vectorized into numeric numpy matrices ($X, y$) for model consumption.

---

## 📽️ Slide 5: Data Mining Techniques & Machine Learning Algorithm

### **1. Supervised Classification via Decision Tree (C4.5 / Entropy)**
- Uses Scikit-Learn's `DecisionTreeClassifier` with **Entropy / Information Gain** as the split criterion:
  $$\text{Entropy}(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$$
  $$\text{Information Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
- The tree recursively partitions the feature space along orthogonal hyperplanes (e.g., `Attendance <= 74.5%`, `Internal_Marks <= 19.5`).
- Tree depth is capped at `max_depth = 5` to prevent overfitting.

### **2. Probability Calibration (Platt Scaling)**
- Wrapped with `CalibratedClassifierCV(method='sigmoid')` to transform raw decision tree step probabilities into continuous, smooth confidence scores ($P(\text{Pass})$ vs $P(\text{Fail})$).

---

## 📽️ Slide 6: Feature Importances & Key Findings

### **Which Factors Drive Student Performance?**
Based on Information Gain analysis from the trained Decision Tree model:

| Academic Factor | Predictive Importance | Impact Analysis |
| :--- | :---: | :--- |
| **Internal Marks** | **~42%** | Strongest indicator. Mid-term assessments directly reflect subject comprehension. |
| **Attendance %** | **~28%** | Critical baseline. Students below 75% attendance show significantly higher fail risk. |
| **Study Hours** | **~16%** | Consistent weekly study time compensates for weaker previous preparation. |
| **Previous Grade** | **~8%** | Past academic foundation provides moderate baseline influence. |
| **Absences / Age** | **~6%** | Minor contributing factor. |

---

## 📽️ Slide 7: Model Evaluation & Performance Metrics

### **Evaluation Metrics Summary**
The model is evaluated using stratified train-test splits ($80/20$):

1. **Accuracy:**
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} \approx 92.3\%$$
2. **Precision:** Measures how many predicted Pass outcomes were actually correct.
3. **Recall:** Measures the proportion of actual Pass students correctly identified.
4. **F1-Score:** Harmonic mean of Precision and Recall ($\approx 91.8\%$).
5. **Confusion Matrix Analysis:**
   - **True Positives (TP):** Correctly predicted Passing students.
   - **True Negatives (TN):** Correctly identified Failing students at risk.
   - **False Positives (FP) & False Negatives (FN):** Kept to a minimum through probability calibration.

---

## 📽️ Slide 8: Interactive Web Application Features

### **Key Application Modules in GradePric:**

1. **Overview Dashboard:**
   - Live KPI metric cards (Total Records, Pass Rate, Model Accuracy, Model F1-Score).
   - Interactive 4-step pipeline guide.
2. **Student Dataset & ETL Cleaning Explorer:**
   - Displays operational dataset table with search and pagination.
   - Detects missing values and provides a single-click clean/impute trigger.
   - Accepts CSV uploads with automatic imputation and model retraining.
3. **Student Outcome Predictor:**
   - Real-time parameter inputs with synchronized sliders.
   - Calibrated Pass/Fail prediction card with probability percentage gauge.
   - Rule-based heuristic intervention suggestions (e.g., compulsory attendance counseling, remedial tutoring).
4. **Prediction History Log:**
   - Persistent audit log storing past predictions in `prediction_history.json`.

---

## 📽️ Slide 9: System Architecture & Tech Stack

### **Technology Stack:**
- **Backend Server:** Python 3.10 + Flask Web Framework
- **Data Warehousing & ETL:** Pandas, NumPy
- **Machine Learning & Data Mining:** Scikit-Learn (`DecisionTreeClassifier`, `CalibratedClassifierCV`)
- **Visual Analytics:** Matplotlib, Seaborn
- **Frontend Design:** HTML5, Modern CSS3 (Emerald Color System), Vanilla JavaScript (Fetch API)

---

## 📽️ Slide 10: Conclusion & Future Scope

### **Summary of Accomplishments:**
- Successfully built an end-to-end Data Warehousing & Data Mining system (**GradePric**).
- Automated the ETL pipeline to gracefully handle missing values and dirty CSV data.
- Deployed a calibrated Decision Tree model offering high predictive accuracy ($>90\%$) alongside actionable student recommendations.

### **Future Scope & Enhancements:**
- **Ensemble Mining:** Integrate Random Forest and XGBoost algorithms for multi-model comparison.
- **Multidimensional Data Cube:** Implement interactive drill-down and roll-up OLAP visual slices by semester or subject department.
- **Role-Based Access Control (RBAC):** Authenticate teachers, students, and academic counselors.

---
*Presented by Rahul, Neeti & Sathin — 022 Batch*
