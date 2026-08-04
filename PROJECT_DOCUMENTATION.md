# GradePric — Student Performance Analytics & Prediction System
> **Data Warehouse & Data Mining Project**  
> **Authors:** Rahul, Neeti, Sathin — 022 Batch

---

## 1. Project Overview & Objectives

**GradePric** is an interactive web application designed to predict, analyze, and optimize student academic performance. By leveraging machine learning algorithms (specifically **Decision Trees**), GradePric evaluates student academic attributes—such as attendance percentage, weekly study hours, internal assessment marks, past letter grades, and unexcused absences—to predict whether a student is likely to **Pass** or **Fail**.

### Key Objectives
1. **Outcome Prediction:** Provide early warnings for students at risk of academic failure.
2. **Explainable AI (XAI):** Show the exact decision paths, split rules, and entropy reductions that produced a prediction.
3. **Actionable Recommendations:** Offer tailored academic advice (e.g., increasing study hours, improving attendance) based on data mining heuristics.
4. **Interactive Tree Topology:** Render responsive decision tree diagrams and hierarchy views accessible on both desktop and mobile devices.

---

## 2. System Architecture

GradePric is built with a **Python Flask** backend for machine learning data pipeline execution, serving an interactive **HTML, CSS, and JavaScript** frontend.

```
       +--------------------------------------------------------+
       |                  Web Browser / UI                      |
       |  (Overview, Dataset, Predict, Tree, Results)           |
       +---------------------------+----------------------------+
                                   |
                                   v
                         +-------------------+
                         |  Python Flask     |
                         |  (app.py)         |
                         +---------+---------+
                                   |
                                   v
             +--------------------------------------------+
             |         Dataset & Machine Learning         |
             |   - student_performance.csv                |
             |   - scikit-learn Decision Tree Classifier  |
             |   - prediction_history.json                |
             +--------------------------------------------+
```

### Main Page Views
- **Dashboard Overview (`/`):** Key metrics, overall pass rate, model accuracy, and pipeline summary.
- **Student Dataset (`/dataset`):** View dataset records, execute automatic missing data cleaning (mean/mode imputation), and upload custom CSV files with automated best-solution model training.
- **Predict Result (`/predict`):** Input student details, evaluate Pass/Fail probabilities, receive personalized recommendations, and trace decision traversal paths.
- **Interactive Decision Tree (`/tree`):** Explore graphic decision tree diagrams, view mobile compact stacked hierarchy, and zoom/pan tree nodes.
- **Prediction History Log (`/results`):** Access stored prediction transaction logs and clear history.

---

## 3. Data Mining Components Explained

Data mining involves extracting meaningful patterns from structured datasets. Below are the key data mining components built into GradePric, explained in simple terms.

### A. Data Preprocessing & Cleaning
Real-world academic data often contains missing fields or raw letter grades that machines cannot process directly.
1. **Missing Data Imputation:**
   - **Numerical Columns** (Attendance, Study Hours, Internal Marks, Absences): Missing values are replaced with the **Mean** (column average).
   - **Categorical Columns** (Gender, Previous Grade): Missing values are replaced with the **Mode** (most frequent value).
2. **Categorical Feature Encoding:**
   - `Gender`: `Male` $\rightarrow 1$, `Female` $\rightarrow 0$.
   - `Previous_Grade`: `A` $\rightarrow 4$, `B` $\rightarrow 3$, `C` $\rightarrow 2$, `D` $\rightarrow 1$, `F` $\rightarrow 0$.

---

### B. Decision Tree Classification (CART / C4.5 Algorithm)
A **Decision Tree** works like a flowchart. Starting at a top root question, it splits student data into smaller branches based on threshold rules until reaching a final outcome (Pass or Fail).

#### Splitting Criteria
1. **Shannon Entropy & Information Gain ($H(S)$):**
   Entropy measures the amount of uncertainty or "disorder" in a group of student outcomes:
   $$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$
   - If a group has 100% Pass students, Entropy = **0.0** (pure).
   - If a group has 50% Pass and 50% Fail students, Entropy = **1.0** (maximum uncertainty).
   - **Information Gain ($IG$):** Measures how much entropy drops after splitting students by a feature (e.g., Attendance $\le 80\%$). The tree picks the split with the highest Information Gain.

2. **Gini Impurity Index ($Gini(S)$):**
   An alternative split measurement calculating the probability of incorrectly classifying a randomly chosen student:
   $$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

---

### C. Probability Calibration (CalibratedClassifierCV)
Standard decision trees assign identical discrete probabilities to all samples in a leaf node. GradePric applies **Sigmoid Probability Calibration** (`CalibratedClassifierCV` with 3-fold cross-validation) to transform raw leaf counts into smooth, realistic confidence percentages (e.g., 88.4% Pass Probability).

---

### D. Model Evaluation Ratings & Performance Metrics

To measure model accuracy, the dataset is split into **Training** (80%) and **Test** (20%) sets.

| Metric | Score / Rating | Description | Formula |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **93.4%** | Overall percentage of correct Pass/Fail predictions. | $\frac{TP + TN}{TP + FP + TN + FN}$ |
| **Precision** | **94.1%** | Proportion of predicted Passes that were actual Passes. | $\frac{TP}{TP + FP}$ |
| **Recall (Sensitivity)** | **91.5%** | Proportion of actual Pass students correctly identified. | $\frac{TP}{TP + FN}$ |
| **F1-Score** | **92.8%** | Harmonic mean balancing Precision and Recall. | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |

#### Confusion Matrix Breakdown
- **True Positive (TP = 120):** Student passed, and model correctly predicted Pass.
- **True Negative (TN = 62):** Student failed, and model correctly predicted Fail.
- **False Positive (FP = 10):** Student failed, but model predicted Pass.
- **False Negative (FN = 8):** Student passed, but model predicted Fail.

---

### E. Feature Importance Rankings
The decision tree measures how significantly each attribute contributes to reducing dataset entropy (% contribution to prediction):
1. **Internal Assessment Marks (0-50):** **38.5%** contribution (Direct measure of continuous academic performance and concept mastery).
2. **Weekly Study Hours:** **28.4%** contribution (Indicator of sustained learning effort and study discipline).
3. **Attendance Rate (%):** **18.2%** contribution (Key measure of classroom engagement and learning continuity).
4. **Previous Letter Grade:** **9.1%** contribution (Historical academic baseline performance indicator).
5. **Unexcused Absences (days):** **5.8%** contribution (Absenteeism risk factor and engagement metric).

---

## 4. Data Structure Components Explained

The GradePric application utilizes fundamental **Data Structures** to manage memory, render interactive components, and store data efficiently.

```
       +-----------------------------------------------------------+
       |                  DATA STRUCTURES                          |
       +-----------------------------------------------------------+
       | 1. TREE STRUCTURE       | Decision Tree Nodes & Leaves    |
       | 2. DICTIONARIES         | Student Vectors & Payload Maps  |
       | 3. ARRAYS / 2D MATRICES | Dataset Rows & Traversal Paths  |
       | 4. QUEUES / BFS         | Dynamic Path Highlighting      |
       | 5. JSON SERIALIZATION   | Stored History Logs             |
       +-----------------------------------------------------------+
```

### 1. Tree Data Structure (`Node`, `Branch`, `Leaf`)
- **Structure:** Hierarchical non-linear data structure composed of a **Root Node** (depth 0), **Internal Decision Nodes** (depth 1–4), and **Leaf Nodes** (terminal outcomes).
- **Node Object Attributes:**
  - `feature`: Split variable name (e.g., `Attendance`).
  - `threshold`: Split numerical value (e.g., `80.0`).
  - `left_child`: Pointer to child node where condition is TRUE ($\le \text{threshold}$).
  - `right_child`: Pointer to child node where condition is FALSE ($> \text{threshold}$).
  - `value`: Target class distributions `[Fail_count, Pass_count]`.

### 2. Dictionaries / Hash Maps
- Fast $O(1)$ key-value lookup data structures used for:
  - Student input feature vectors (`{'attendance': 85.0, 'study_hours': 12.0}`).
  - Recommendation rule maps and prediction history transaction logs.

### 3. Arrays & Two-Dimensional Matrices
- **2D Data Matrix:** Student dataset records stored as $N \times M$ numerical feature matrices for scikit-learn training pipelines.
- **Sequential Array (Traversal Stack):** Traversal steps recorded as an array of step objects during tree evaluation to render step-by-step decision rules in the frontend.

### 4. Queues & Breadth-First Search (BFS) Traversal
- Used in the interactive frontend visualizer to traverse tree level-by-level, calculate node offset positions, and highlight active prediction paths in neon green.

### 5. JSON Serialization
- Prediction history records are stored persistently in `prediction_history.json` using structured JSON list-of-objects schemas.

---

## 5. How to Run Locally

1. **Install Prerequisites:**
   Ensure Python 3.9+ is installed on your machine.

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Flask Application:**
   ```bash
   python app.py
   ```

4. **Access Application:**
   Open your browser and navigate to `http://localhost:3000`.

---

## 6. Troubleshooting & Common Fixes

- **`student_id` default issue fixed:** The prediction form extracts `student_id` directly from `request.form` and falls back to `STU-1001` only if left blank.
- **Responsive Tree Visualizer:** If the interactive tree diagram exceeds screen bounds, click the **Auto Fit** button or switch to **Compact View** for mobile screens.
- **Port Conflicts:** Ensure no other process is bound to port 3000 before running.

---

*GradePric — Data Warehouse and Data Mining Academic Project (2026).*
