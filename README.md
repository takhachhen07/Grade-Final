# Data Warehousing & Data Mining Suite 🎓

An interactive Data Warehousing and Data Mining platform supporting ETL data processing, Operational Data Store staging, OLAP Data Cubes (Slice, Dice, Roll-Up, Drill-Down), Association Rule Mining (Apriori Algorithm), and K-Means Cluster Analysis.

> 📖 **Detailed Presentation Outline:** For a slide-by-slide presentation breakdown, see [presentation.md](presentation.md).  
> 📖 **Detailed Technical Documentation:** For comprehensive architectural details, formulas, and algorithms, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

---

## 📌 Project Overview

This project provides an intuitive web platform for educational analytics and data mining. It transforms raw student engagement metrics into actionable insights through 4 core modules:

1. **ETL & Operational Data Store (ODS):** Ingest raw dataset records, execute automated mean/mode imputation for missing values, and stage records in the ODS (`uploaded_dataset.csv`).
2. **OLAP Data Cube Engine:** Multidimensional query matrix supporting Slice, Dice, Roll-Up, and Drill-Down operations across Attendance, Study Hours, Grades, and Gender.
3. **Association Rule Mining (Apriori Algorithm):** Discretizes attributes into transaction itemsets and mines frequent itemsets and high-confidence association rules ($IF \rightarrow THEN$).
4. **Cluster Analysis (K-Means Clustering):** Unsupervised student profile segmentation into $K$ behavioral clusters (High-Performing Achievers, At-Risk Academic Warnings, Dedicated High-Effort) with interactive scatter plot maps.

---

## 🛠️ Technologies Used
* **Programming Language:** Python 3.11+
* **Backend Web Framework:** Flask 3.0+
* **Data Analytics & Warehousing:** Pandas, NumPy
* **Data Mining Algorithms:**
  * Custom Apriori Itemset Mining (`utils/association_rules.py`)
  * Custom Vectorized K-Means Clustering (`utils/clustering.py`)
  * Multidimensional OLAP Aggregation Engine (`utils/olap_engine.py`)
* **Frontend UI & Styling:** HTML5, Responsive CSS3, SVG Data Canvas, FontAwesome

---

## 📊 Operational Data Store Schema

| Feature Name | Type | Description / Range |
| :--- | :--- | :--- |
| `Student_ID` | String | Unique Student Identifier (e.g., STU-1001) |
| `Gender` | Categorical | Male / Female |
| `Age` | Integer | Student Age (17 to 23) |
| `Attendance` | Float | Class Attendance Percentage (0.0% to 100.0%) |
| `Study_Hours` | Float | Weekly Study Time (1.0 to 30.0 hours) |
| `Internal_Marks`| Integer | Mid-term/Internal score (0 to 50) |
| `Previous_Grade` | Categorical | Past Letter Grade (A, B, C, D, F) |
| `Absences` | Integer | Total Days Absent (0 to 20) |
| `Result` | Categorical | Target Outcome: **Pass** or **Fail** |

---

## 🚀 How to Run Locally

### Step-by-Step Execution Guide

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask Application:**
   ```bash
   python app.py
   ```

3. **Access the Application:**
   Open your browser at `http://localhost:3000`.

---

## 📁 Directory Structure
```
├── app.py                      # Flask Application Entry Point & API Routes
├── requirements.txt            # Python Dependencies
├── uploaded_dataset.csv         # Operational Data Store (ODS) File
├── templates/
│   ├── base.html              # Base Layout & Sidebar Navigation
│   ├── index.html             # Overview Dashboard
│   ├── dataset.html           # Dataset Explorer & ETL Data Cleaning
│   ├── olap.html              # OLAP Data Cube Multidimensional Analysis
│   ├── association_rules.html # Apriori Association Rule Mining
│   └── clustering.html        # K-Means Cluster Analysis
├── utils/
│   ├── data_processor.py      # ETL Data Ingestion & Cleaning
│   ├── olap_engine.py         # OLAP Cube Slicing/Dicing Matrix Engine
│   ├── association_rules.py  # Apriori Itemset & Rule Generator
│   └── clustering.py         # Vectorized K-Means Segmentation
├── presentation.md            # Slide-by-Slide Presentation Guide
└── PROJECT_DOCUMENTATION.md   # Comprehensive Technical Documentation
```

---
*Data Warehousing & Data Mining Suite — Presented by Rahul, Neeti, Sathin (Batch 022).*
