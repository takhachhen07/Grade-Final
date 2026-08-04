# 🎓 Data Warehousing & Data Mining Suite — Project Presentation
> **Project Title:** Educational Data Warehousing & Multidimensional Mining Suite  
> **Course:** Data Warehouse and Data Mining  
> **Team Members:** Rahul, Neeti & Sathin — **022 Batch**  
> **Frameworks:** Python Flask, Pandas, NumPy, Association Rule Mining (Apriori), K-Means Clustering, OLAP Data Cube Engine  

---

## 📽️ Slide 1: Title & Project Overview

### **Data Warehousing & Data Mining Suite for Academic Analytics**
- **Presenter Team:** Rahul, Neeti & Sathin (Batch 022)
- **Domain:** Educational Data Mining (EDM) & Data Warehousing (DW)
- **Objective:** Deliver an intuitive, user-friendly web suite that combines an Operational Data Store (ODS), automated ETL data cleaning, an OLAP Data Cube engine (Slice, Dice, Roll-Up, Drill-Down), Association Rule Mining via the Apriori Algorithm, and K-Means Cluster Analysis.

---

## 📽️ Slide 2: Project Architecture & Core Modules

### **4 Core Architectural Modules**

```
+-----------------------------------------------------------------------------------+
|                        DATA WAREHOUSING & MINING SUITE                             |
+--------------------------+--------------------------+-----------------------------+
| 1. ETL & ODS Store       | Extract, Clean (Impute), & Stage Data into ODS Database   |
| 2. OLAP Data Cube        | Multidimensional Slice, Dice, Roll-Up, & Drill-Down Matrix |
| 3. Association Rules     | Apriori Frequent Itemsets & Rule Mining (IF -> THEN)      |
| 4. K-Means Clustering    | Behavioral Student Profile Segmentation & Scatter Maps    |
+--------------------------+--------------------------+-----------------------------+
```

1. **ETL & Operational Data Store (ODS):** Ingests raw CSV datasets, handles flexible header mapping, applies mean/mode statistical imputation for missing values, and stages clean records in `uploaded_dataset.csv`.
2. **OLAP Data Cube Engine:** Multidimensional query processing across dimensions like Previous Grade, Attendance Tier, Study Hours Tier, and Gender with interactive Slice & Dice filters.
3. **Association Rule Mining (Apriori):** Discretizes student attributes into itemsets and mines high-confidence association rules with adjustable Minimum Support and Minimum Confidence thresholds.
4. **Cluster Analysis (K-Means):** Groups student profiles into $K$ behavioral clusters (e.g., High-Performing Achievers, At-Risk Academic Warnings) using Euclidean distance and standard scaling.

---

## 📽️ Slide 3: Module 1 — ETL (Extract, Transform, Load) & ODS

### **1. Extract (E):**
- Ingests raw student records from uploaded CSV files or web forms.
- Performs flexible header mapping (e.g., `studyhours` $\rightarrow$ `Study_Hours`, `test_score` $\rightarrow$ `Internal_Marks`).

### **2. Transform (T):**
- **Missing Value Imputation:**
  - *Numerical Features* (`Attendance`, `Study_Hours`, `Internal_Marks`, `Absences`, `Age`): Replaced using **Mean Imputation**:
    $$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$
  - *Categorical Features* (`Gender`, `Previous_Grade`): Replaced using **Mode Imputation** (most frequent value).
- **Domain Bound Clipping:** Bounds numeric attributes to realistic physical ranges (e.g., Attendance constrained to $[0.0, 100.0]\%$, Internal Marks bounded to $[0, 50]$).

### **3. Load (L):**
- Staged into the Operational Data Store (`uploaded_dataset.csv`) for analytical processing across OLAP, Apriori, and K-Means modules.

---

## 📽️ Slide 4: Module 2 — OLAP Data Cube Engine

### **Multidimensional OLAP Operations**
- **Data Cube Aggregations:** Evaluates measures including Cell Count, Pass Count, Fail Count, Pass Rate (%), Mean Attendance, Mean Study Hours, Mean Internal Marks, and Absences.
- **OLAP Operations Implemented:**
  1. **Slice:** Filters the data cube along a single dimension (e.g., `Slice: Gender = Male` or `Slice: Result = Pass`).
  2. **Dice:** Constructs a sub-cube by selecting specific values across multiple dimensions simultaneously.
  3. **Roll-Up:** Aggregates metrics up to higher-level dimensional hierarchies (e.g., grouping by `Previous_Grade`).
  4. **Drill-Down:** Disaggregates metrics down to finer granularities (e.g., adding `Attendance_Tier` or `Study_Tier` breakdown).

---

## 📽️ Slide 5: Module 3 — Association Rule Mining (Apriori Algorithm)

### **1. Discretization into Transaction Itemsets**
- Converts continuous student attributes into discrete categorical transaction items:
  - Attendance: `High_Attendance (≥80%)` vs `Low_Attendance (<80%)`
  - Study Hours: `High_Study_Hours (≥10h)` vs `Low_Study_Hours (<10h)`
  - Internal Marks: `High_Internal_Marks (≥30/50)` vs `Low_Internal_Marks (<30/50)`
  - Outcome: `Outcome_Pass` vs `Outcome_Fail`

### **2. Apriori Metrics & Formulas**
- **Support:** Percentage of transactions containing itemset $X \cup Y$:
  $$\text{Support}(X \Rightarrow Y) = \frac{\text{Count}(X \cup Y)}{N}$$
- **Confidence:** Conditional probability that $Y$ occurs given $X$:
  $$\text{Confidence}(X \Rightarrow Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$$
- **Lift Score:** Measures how much more often $X$ and $Y$ occur together than expected if independent:
  $$\text{Lift}(X \Rightarrow Y) = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)}$$

---

## 📽️ Slide 6: Module 4 — Cluster Analysis (K-Means Clustering)

### **Unsupervised Behavioral Student Segmentation**
- **Feature Vector:** Standardized vector of student engagement metrics: $[ \text{Attendance}, \text{Study\_Hours}, \text{Internal\_Marks}, \text{Absences} ]$.
- **Distance Metric:** Euclidean Distance in normalized feature space:
  $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
- **Algorithm Iteration:**
  1. Initialize $K$ cluster centroids.
  2. Assign each student profile to the nearest centroid.
  3. Recalculate centroids as the mean vector of assigned profiles.
  4. Repeat until convergence.
- **Discovered Student Clusters:**
  - **Cluster 1: High-Performing Achievers** (High attendance $\ge 80\%$, High internal marks $\ge 30/50$).
  - **Cluster 2: At-Risk Academic Warning** (Low attendance $< 70\%$, Low internal marks $< 22/50$).
  - **Cluster 3: Dedicated High-Effort** (High weekly study hours $\ge 12$h).

---

## 📽️ Slide 7: User Experience & Simplified Interface Design

### **Designed for Ease of Use**
- **Single-Click Workflow:** Instant ETL data cleaning and single-slider hyperparameter adjustments.
- **Clear Navigation:** Clean sidebar separating Data Warehouse & ODS from Data Mining Suite.
- **Visual Feedback:** Interactive scatter plot maps, clear metric cards, and color-coded status badges (`Pass`, `Fail`, `High/Low`).
- **Zero Configuration:** Pre-loaded operational dataset allowing immediate execution of OLAP queries, Apriori mining, and K-Means clustering.

---

## 📽️ Slide 8: Technical Stack & Implementation Details

### **Technology Stack:**
- **Backend Framework:** Python 3.11 + Flask Web Application Framework
- **Data Warehousing & Analytics:** Pandas, NumPy
- **Data Mining Algorithms:**
  - Custom Apriori Itemset Mining Engine (`utils/association_rules.py`)
  - Custom Vectorized K-Means Clustering (`utils/clustering.py`)
  - Multidimensional OLAP Aggregation Engine (`utils/olap_engine.py`)
- **Frontend UI & Visuals:** HTML5, Modern Responsive CSS3 (Tailwind-inspired CSS design), SVG Data Canvas, FontAwesome Icons

---

## 📽️ Slide 9: Summary & Future Scope

### **Key Accomplishments:**
1. Built a complete, functional Data Warehouse and Data Mining platform.
2. Implemented automated ETL preprocessing with missing data imputation and Operational Data Store staging.
3. Created an interactive OLAP Data Cube supporting Slice, Dice, Roll-Up, and Drill-Down operations.
4. Integrated Apriori Association Rule Mining and K-Means Clustering with live interactive parameter controls.

### **Future Scope:**
- Multi-dimensional OLAP cube export to Excel / CSV.
- Automated alert triggers for student profiles falling into At-Risk clusters.

---
*Presented by Rahul, Neeti & Sathin — 022 Batch*
