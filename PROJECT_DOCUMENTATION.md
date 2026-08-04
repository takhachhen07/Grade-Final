# Data Warehousing & Data Mining Suite — System Documentation
> **Data Warehouse & Data Mining Project**  
> **Authors:** Rahul, Neeti, Sathin — 022 Batch

---

## 1. Project Overview & Objectives

The **Data Warehousing & Data Mining Suite** is an interactive web platform designed for educational data management, multidimensional analytics, association rule mining, and student profile clustering.

By staging operational student metrics into an **Operational Data Store (ODS)**, the application enables educators and data analysts to clean raw data via **ETL pipelines**, perform **OLAP Data Cube** slicing and dicing, uncover hidden behavioral patterns using the **Apriori Algorithm**, and segment student populations with **K-Means Clustering**.

### Key System Features
1. **ETL & Operational Data Store (ODS):** Ingest raw dataset records, execute automated statistical mean/mode imputation for missing values, and stage clean records.
2. **OLAP Data Cube Engine:** Multidimensional query processing with interactive Slice, Dice, Roll-Up, and Drill-Down controls.
3. **Association Rule Mining (Apriori Algorithm):** Mine frequent itemsets and high-confidence association rules ($IF \rightarrow THEN$) connecting student habits with academic outcomes.
4. **Cluster Analysis (K-Means Clustering):** Unsupervised segmentation of student profiles into distinct behavioral clusters with interactive distribution map scatter plots.

---

## 2. System Architecture

The application is built on a **Python Flask** backend executing data warehousing and mining algorithms, serving a responsive, accessible **HTML5, CSS3, and JavaScript** frontend.

```
       +--------------------------------------------------------+
       |                  Web Browser / UI                      |
       |  (Overview, Dataset/ETL, OLAP Cube, Apriori, K-Means)  |
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
             |         Dataset & Analytics Engines        |
             |   - uploaded_dataset.csv (ODS Storage)     |
             |   - utils/data_processor.py (ETL)          |
             |   - utils/olap_engine.py (OLAP Cube)       |
             |   - utils/association_rules.py (Apriori)   |
             |   - utils/clustering.py (K-Means)          |
             +--------------------------------------------+
```

---

## 3. Core Modules & Mathematical Principles

### A. Module 1: ETL Pipeline & Operational Data Store (ODS)
1. **Extract:** Flexible header mapping ingests uploaded CSV files into standard feature names (`Attendance`, `Study_Hours`, `Internal_Marks`, `Previous_Grade`, `Absences`, `Gender`, `Age`, `Result`).
2. **Transform (Data Cleaning):**
   - **Numerical Imputation:** Missing numerical values are replaced with the mean:
     $$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$
   - **Categorical Imputation:** Missing categorical fields are replaced with the mode (most frequent item).
   - **Range Enforcement:** Constrains values within valid physical bounds (e.g., Attendance $0-100\%$, Internal Marks $0-50$).
3. **Load:** Cleansed dataset is stored in `uploaded_dataset.csv` for downstream mining modules.

---

### B. Module 2: OLAP Data Cube Engine
The OLAP engine constructs a multidimensional data cube over student attributes:
- **Dimensions:** Previous Grade, Attendance Tier (Low/Moderate/High), Study Hours Tier (Low/Moderate/High), Gender.
- **Measures Evaluated:** Cell Record Count, Pass Count, Fail Count, Pass Rate (%), Mean Attendance, Mean Study Hours, Mean Internal Marks, Mean Absences.
- **Operations:**
  - **Slice:** Filter the cube on a single dimension value (e.g., `Slice: Gender = Male`).
  - **Dice:** Select a sub-cube by applying multi-dimensional filters.
  - **Roll-Up / Drill-Down:** Aggregate or disaggregate metrics across selected dimension hierarchies.

---

### C. Module 3: Association Rule Mining (Apriori Algorithm)
Transforms continuous attributes into itemsets and applies the Apriori algorithm to discover frequent itemsets and rules ($X \Rightarrow Y$):
- **Discretized Items:** `High_Attendance (≥80%)`, `High_Study_Hours (≥10h)`, `High_Internal_Marks (≥30/50)`, `Outcome_Pass`, `Outcome_Fail`.
- **Formulas:**
  - **Support:** $\text{Support}(X \Rightarrow Y) = \frac{\text{Count}(X \cup Y)}{N}$
  - **Confidence:** $\text{Confidence}(X \Rightarrow Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$
  - **Lift:** $\text{Lift}(X \Rightarrow Y) = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)}$

---

### D. Module 4: K-Means Cluster Analysis
Performs unsupervised learning to group student profiles into $K$ clusters based on standardized features ($\text{Attendance}, \text{Study\_Hours}, \text{Internal\_Marks}, \text{Absences}$):
- **Distance Metric:** Euclidean Distance:
  $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
- **Clusters Profiles Generated:**
  - **Cluster 1: High-Performing Achievers** (High Attendance & Internal Marks).
  - **Cluster 2: At-Risk Academic Warning** (Low Attendance & Internal Marks).
  - **Cluster 3: Dedicated High-Effort** (High Study Hours).

---

## 4. How to Run the Application

1. **Install Prerequisites:**
   Ensure Python 3.9+ is installed.

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Application:**
   ```bash
   python app.py
   ```

4. **Access Web Suite:**
   Open browser at `http://localhost:3000`.

---
*Data Warehousing & Data Mining Project — Rahul, Neeti, Sathin (Batch 022).*
