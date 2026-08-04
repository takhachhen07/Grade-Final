# Data Warehousing & Data Mining Suite
> **Course:** Data Warehouse & Data Mining  
> **Team Members:** Rahul, Neeti & Sathin — **022 Batch**

---

## 📌 1. Suite Overview

An end-to-end Data Warehousing and Data Mining platform built using **Python Flask**, **Pandas**, and **NumPy**. It implements ETL data preprocessing, OLAP multidimensional data cubes, Apriori association rule mining, and K-Means cluster analysis.

---

## 🏬 2. Data Warehouse (DW) Modules

### A. Operational Data Store (ODS)
- `uploaded_dataset.csv` acts as the central staging database holding student academic metrics.

### B. ETL (Extract, Transform, Load) Pipeline
1. **Extract**: Ingests raw data from CSV uploads or forms.
2. **Transform**: Missing numerical values are filled with the **Mean**; categorical values with the **Mode**. Range clipping bounds values to valid limits.
3. **Load**: Saves cleaned data to the ODS for analytics modules.

### C. OLAP Data Cube Engine
- Supports **Slice**, **Dice**, **Roll-Up**, and **Drill-Down** multidimensional analytical operations across Attendance Tiers, Study Hours Tiers, Grades, and Gender.

---

## ⛏️ 3. Data Mining (DM) Modules

### A. Association Rule Mining (Apriori)
- Discretizes attributes into transaction itemsets and discovers high-confidence association rules ($IF \rightarrow THEN$) with customizable Minimum Support and Minimum Confidence controls.

### B. Cluster Analysis (K-Means)
- Groups student profiles into $K$ behavioral clusters (e.g., High-Performing Achievers, At-Risk Academic Warnings, Dedicated High-Effort) with interactive scatter plot maps.

---
*Presented by Rahul, Neeti & Sathin — 022 Batch*
