# Student Performance Prediction Using Decision Tree 🎓

A comprehensive Data Mining project predicting whether a student will **Pass** or **Fail** based on academic parameters using a **Decision Tree Classifier** and a **Flask / Node.js** web application.

> 📖 **Detailed Documentation:** For a complete breakdown of all Data Mining concepts, Data Structures, Machine Learning algorithms, and mathematical formulas, please see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

---

## 📌 Project Overview
Predicting student academic performance enables early identification of at-risk students, allowing educators and institutions to provide timely interventions. This project implements an end-to-end Machine Learning pipeline utilizing Data Mining concepts, from raw CSV data ingestion to preprocessing, supervised model training, dynamic metric evaluation, and an interactive web interface.

---

## 🎯 Objectives
* **Data Ingestion & Cleaning:** Load dataset and handle missing values automatically using mean/median/mode imputation.
* **Categorical Encoding:** Encode categorical student attributes (`Gender`, `Previous_Grade`, `Result`) into numerical vectors.
* **Supervised Learning:** Train a **Decision Tree Classifier** to classify students into Pass or Fail classes.
* **Performance Evaluation:** Dynamically compute Model Accuracy, Confusion Matrix, Precision, Recall, and F1-Score.
* **Visual Analytics:** Generate visual plots for feature distribution, feature importance, and confusion matrix heatmaps.
* **Interactive Prediction Form:** Provide a simple, web-based form for predicting individual student outcomes instantly with personalized academic feedback.

---

## 🛠️ Technologies Used
* **Programming Language:** Python 3.10+
* **Backend Web Framework:** Flask 3.0+
* **Data Mining & Analytics:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (`DecisionTreeClassifier`, `train_test_split`, `metrics`)
* **Data Visualization:** Matplotlib, Seaborn
* **Model Serialization:** Joblib (`model.pkl`)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript

---

## 📊 Dataset Description (`student_performance.csv`)
The project includes a realistic dataset containing **500 student records** with 9 key features:

| Feature Name | Type | Description / Range |
| :--- | :--- | :--- |
| `Student_ID` | String | Unique Identifier (e.g., STU1001) |
| `Gender` | Categorical | Male / Female |
| `Age` | Integer | Student Age (17 to 23) |
| `Attendance` | Float | Class Attendance Percentage (0.0% to 100.0%) |
| `Study_Hours` | Float | Daily Study Time (1.0 to 12.0 hours) |
| `Internal Marks(20-50)`| Integer | Mid-term/Internal assessment score (20 to 50) |
| `Previous_Grade` | Categorical | Historical academic grade (A, B, C, D, F) |
| `Absences` | Integer | Total days absent (0 to 20) |
| `Result` | Categorical | Target Outcome: **Pass** or **Fail** |

---

## ⚙️ Data Preprocessing & Pipeline
1. **Missing Value Imputation:**
   - Numerical columns (`Attendance`, `Study_Hours`, `Internal Marks(20-50)`, `Age`, `Absences`) missing values filled with column median.
   - Categorical columns (`Gender`, `Previous_Grade`) missing values filled with mode.
2. **Label Encoding:**
   - `Gender`: Male = 0, Female = 1
   - `Previous_Grade`: A = 4, B = 3, C = 2, D = 1, F = 0
   - `Result`: Pass = 1, Fail = 0
3. **Train-Test Split:**
   - 80% Training Set, 20% Testing Set with stratified sampling.

---

## 🌳 Decision Tree Algorithm Explanation
The **Decision Tree Classifier** is a non-parametric supervised learning algorithm used for classification. It splits data recursively based on feature thresholds that maximize class purity, measured using **Gini Impurity**:

$$Gini(D) = 1 - \sum_{i=1}^{k} p_i^2$$

Where $p_i$ is the probability of an item belonging to class $i$. The feature with the highest **Information Gain** (or lowest Gini Impurity) becomes the split node.

---

## 🚀 How to Run Locally

### Prerequisites
* Python 3.8 or higher installed on your machine.
* `pip` package manager.

### Step-by-Step Execution Guide

1. **Clone or Download the Repository:**
   ```bash
   git clone https://github.com/your-username/student-performance-prediction.git
   cd student-performance-prediction
   ```

2. **Install Required Python Packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Dataset (Optional / First Time):**
   ```bash
   python generate_data.py
   ```

4. **Run the Flask Web Application:**
   ```bash
   python app.py
   ```

5. **Access the Web Application:**
   Open your web browser and navigate to:
   ```
   http://127.0.0.1:3000
   ```

---

## 📁 Project Directory Structure
```
student-performance-prediction/
│
├── app.py                     # Flask Web Backend & ML Pipeline
├── requirements.txt           # Python Dependencies
├── generate_data.py           # Script generating student_performance.csv
├── student_performance.csv    # 500-record dataset
├── model.pkl                  # Serialized Decision Tree Model
├── templates/
│   ├── index.html             # Home Page
│   ├── dataset.html           # Dataset Explorer, Data Cleaning & Auto-Train CSV Upload
│   ├── predict.html           # Student Outcome Predictor Form
│   ├── tree.html              # Interactive Decision Tree Visualizer
│   └── results.html           # Visual Analytics & Reports
├── static/
│   ├── css/
│   │   └── style.css          # Custom Responsive Styling
│   ├── js/
│   │   └── main.js            # Client-side Interactions
│   └── images/                # Matplotlib Generated Charts
├── PROJECT_REPORT.md          # College Submission Project Report
├── PRESENTATION.md            # 10-Slide PPT Presentation Outline
└── README.md                  # Project Documentation
```

---

## 🔮 Future Enhancements
* Incorporate advanced ensemble algorithms like **Random Forest** and **XGBoost** for comparative benchmarking.
* Integrate hyperparameter optimization using `GridSearchCV`.
* Add student performance trend analysis over multiple semesters.
* Implement user authentication for teachers and administrators.

---

## 📚 References
1. Quinlan, J. R. (1986). *Induction of decision trees*. Machine Learning, 1(1), 81-106.
2. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques*. Morgan Kaufmann.
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
