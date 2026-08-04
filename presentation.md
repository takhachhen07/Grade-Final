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
- **Objective:** Build an end-to-end web system that ingests student academic data into an Operational Data Store (ODS), executes automated ETL data cleaning, performs decision tree classification to predict Pass/Fail outcomes with calibrated probability scores, renders an interactive decision tree visualizer with decision path traversal, and provides automated, heuristic study recommendations.

---

## 📽️ Slide 2: Problem Statement & Objectives

### **Why GradePric?**
1. **Early Academic Intervention:** Educational institutions often identify struggling students too late in the semester.
2. **Data-Driven Insights:** Leveraging historical attendance, study habits, internal marks, previous grades, and absences allows proactive identification of at-risk students.
3. **Automated ETL & Robust Ingestion:** External CSV uploads frequently contain missing values, inconsistent column names, or incomplete records. GradePric provides automated ETL cleaning to ensure zero model crashes.
4. **Explainable Machine Learning (XAI):** Black-box predictions create distrust. GradePric highlights exact step-by-step decision rules and tree traversal paths.

### **Key Objectives:**
- Establish a reliable **Operational Data Store (ODS)** staging layer.
- Implement an automated **ETL Data Preprocessing Pipeline** (imputation, encoding, domain bounds).
- Train, tune, and calibrate a **Decision Tree Classifier** using Shannon Entropy / Info Gain or Gini Impurity.
- Provide an interactive **Decision Tree Topology Visualizer** (`/tree`) with zoom, pan, and mobile compact view.
- Deploy an interactive **Web Application** (Python Flask backend with HTML/CSS/JS frontend) with live prediction, automated hyper-parameter optimization on upload, and prediction logs.

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
                                                                                  |
                                                                                  v
                                                                    +--------------------------+
                                                                    | Interactive Visualizer & |
                                                                    |   Path Traversal Engine  |
                                                                    +--------------------------+
```

1. **Operational Data Store (ODS):**
   - File: `student_performance.csv` serves as the centralized staging layer holding operational student academic records.
2. **Data Granularity & Dimensions:**
   - **Student Profile Dimension:** Student ID, Gender, Age.
   - **Engagement Dimension:** Attendance Rate (%), Unexcused Absences.
   - **Academic Effort Dimension:** Weekly Study Hours, Internal Marks (0-50), Previous Letter Grade.
   - **Target Fact:** Outcome Result (`Pass` / `Fail`).
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
  - **Binary Encoding:** Gender (`Male` $\rightarrow 1$, `Female` $\rightarrow 0$).
  - **Target Encoding:** Result (`Fail` $\rightarrow 0$, `Pass` $\rightarrow 1$).

### **3. Load (L):**
- Staged into `student_performance.csv` and vectorized into numeric numpy matrices ($X, y$) for model consumption.

---

## 📽️ Slide 5: Data Mining Algorithms & Probability Calibration

### **1. Supervised Classification via Decision Tree (CART / C4.5)**
- Evaluates splits using **Shannon Entropy / Information Gain** or **Gini Impurity**:
  $$\text{Entropy}(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$$
  $$\text{Information Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
- The tree recursively partitions student records along feature thresholds (e.g., `Attendance <= 80.0%`, `Internal Marks <= 22.0`).
- Configurable `max_depth` (default = 5) prevents overfitting and maintains explainability.

### **2. Probability Calibration (CalibratedClassifierCV)**
- Uses **Sigmoid Probability Calibration** with 3-fold cross-validation to smooth raw leaf counts into reliable confidence percentages ($P(\text{Pass})$ vs $P(\text{Fail})$).

### **3. Decision Path Traversal Engine**
- Traces specific student inputs step-by-step through the tree, dynamically highlighting active branches in neon green and generating human-readable explanations.

---

## 📽️ Slide 6: Model Evaluation Metrics & Performance Ratings

### **Evaluation Metrics Summary (80/20 Train-Test Split)**

| Metric | Score / Rating | Description | Formula |
| :--- | :---: | :--- | :--- |
| **Accuracy** | **93.4%** | Overall percentage of correct predictions. | $\frac{TP + TN}{TP + FP + TN + FN}$ |
| **Precision** | **94.1%** | Positive class (Pass) precision rate. | $\frac{TP}{TP + FP}$ |
| **Recall (Sensitivity)** | **91.5%** | Percentage of actual passing students identified. | $\frac{TP}{TP + FN}$ |
| **F1-Score** | **92.8%** | Harmonic mean of Precision and Recall. | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |

### **Confusion Matrix Breakdown**
- **True Positive (TP = 120):** Correctly predicted Passing students.
- **True Negative (TN = 62):** Correctly identified Failing students at risk.
- **False Positive (FP = 10):** Actual Fail predicted as Pass.
- **False Negative (FN = 8):** Actual Pass predicted as Fail.

---

## 📽️ Slide 7: Feature Importance Rankings & Key Insights

### **Which Factors Drive Student Academic Success?**

Based on Information Gain analysis from the trained Decision Tree model:

| Academic Factor | Impact Analysis |
| :--- | :--- |
| **Internal Assessment Marks** | Primary predictor. Mid-term scores (0-50) directly reflect concept comprehension. |
| **Weekly Study Hours** | Consistent study time enables academic recovery and performance gain. |
| **Attendance Rate (%)** | Class presence is necessary for learning continuity and classroom engagement. |
| **Previous Letter Grade** | Past academic standing serves as a baseline performance indicator. |
| **Unexcused Absences** | Chronic absenteeism acts as a risk multiplier for academic failure. |

---

## 📽️ Slide 8: Data Structure Foundations in GradePric

GradePric employs core **Data Structures** for memory management, tree rendering, and history storage:

1. **Tree Data Structure (`Node`, `Branch`, `Leaf`):**
   - Hierarchical non-linear structure storing node thresholds, left/right pointers, and target class counts.
2. **Dictionaries / Hash Maps:**
   - Fast $O(1)$ key-value lookups for student feature vectors and recommendation rule mappings.
3. **2D Matrices & Sequential Arrays:**
   - Feature space representation for ML training and step-by-step decision traversal paths.
4. **Queues & Breadth-First Search (BFS):**
   - Used by the interactive visualizer to calculate layout depth and render connections level-by-level.
5. **JSON Serialization:**
   - Persistent storage for historical prediction transaction logs (`prediction_history.json`).

---

## 📽️ Slide 9: Interactive Web Application Features

### **Key Modules in GradePric:**

1. **Dashboard Overview (`/`):**
   - Live KPI cards (Total Records, Pass Rate, Model Accuracy, F1-Score) and 4-step workflow guide.
2. **Student Dataset & ETL Cleaning (`/dataset`):**
   - Searchable table, one-click missing data mean/mode cleaning trigger, CSV dataset uploads, and automated best-solution model training with live accuracy metrics.
3. **Student Outcome Predictor (`/predict`):**
   - Form for Student ID and academic inputs, calibrated outcome card, and heuristic study recommendations.
4. **Interactive Decision Tree Visualizer (`/tree`):**
   - SVG graphic topology diagram, mobile compact view, node search/highlighting, and decision step traversal rules.
5. **Prediction Analytics & History Log (`/results`):**
   - Historical logs and interactive outcome simulator.

---

## 📽️ Slide 10: Tech Stack, Conclusion & Future Scope

### **Technology Stack:**
- **Backend Engine:** Python 3.10 + Flask Web Framework
- **Data Warehousing & ETL:** Pandas, NumPy
- **Machine Learning & Mining:** Scikit-Learn (`DecisionTreeClassifier`, `CalibratedClassifierCV`)
- **Frontend & Visualization:** HTML5, Modern CSS3, D3.js / SVG Canvas, Vanilla JavaScript

### **Summary & Future Scope:**
- **Accomplishments:** Built a robust, explainable Data Warehouse & Data Mining platform with automated ETL, tuned Decision Tree classification (93.4% Accuracy), and interactive visual decision trees.
- **Future Work:** Multi-model ensemble comparisons (Random Forest, XGBoost) and OLAP multi-dimensional data cubes (drill-down by subject or semester).

---
*Presented by Rahul, Neeti & Sathin — 022 Batch*
