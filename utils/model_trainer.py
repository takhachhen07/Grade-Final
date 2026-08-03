import os
import pickle
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils.data_processor import clean_dataframe, encode_dataframe

FEATURE_COLUMNS = ['Gender_Encoded', 'Age', 'Attendance', 'Study_Hours', 'Internal_Marks', 'Grade_Encoded', 'Absences']
FEATURE_NAMES = ['Gender', 'Age', 'Attendance (%)', 'Study Hours (hrs/wk)', 'Internal Marks (0-50)', 'Previous Grade', 'Absences (days)']

def train_decision_tree(df, criterion='entropy', max_depth=5, test_size=0.2, model_path='model.pkl'):
    """
    Trains a Decision Tree Classifier using Information Theory (Entropy/Information Gain or Gini),
    applies Sigmoid Probability Calibration via CalibratedClassifierCV, evaluates performance metrics,
    and exports tree rules and model artifacts.
    """
    # ETL pre-processing
    df_clean = clean_dataframe(df)
    df_encoded = encode_dataframe(df_clean)

    X = df_encoded[FEATURE_COLUMNS]
    y = df_encoded['Result_Encoded']

    # Train / Test Split
    class_counts = y.value_counts()
    use_stratify = len(class_counts) > 1 and class_counts.min() >= 2

    if len(y) < 10:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if use_stratify else None
            )
        except Exception:
            X_train, X_test, y_train, y_test = X, X, y, y

    # Decision Tree Classifier
    dt = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=42
    )
    dt.fit(X_train, y_train)

    # Probability Calibration using CalibratedClassifierCV
    train_class_counts = y_train.value_counts()
    if len(train_class_counts) > 1 and train_class_counts.min() >= 3:
        calibrated_dt = CalibratedClassifierCV(estimator=dt, method='sigmoid', cv=3)
        calibrated_dt.fit(X_train, y_train)
    else:
        try:
            calibrated_dt = CalibratedClassifierCV(estimator=dt, method='sigmoid', cv='prefit')
            calibrated_dt.fit(X_train, y_train)
        except Exception:
            calibrated_dt = dt

    # Model evaluation on Test set
    y_pred = calibrated_dt.predict(X_test)
    y_proba = calibrated_dt.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    # Decision Tree Rules Extraction
    tree_rules = export_text(dt, feature_names=FEATURE_NAMES)

    # Feature Importance
    importances = dict(zip(FEATURE_NAMES, dt.feature_importances_))

    # Save artifact
    artifact = {
        'model': calibrated_dt,
        'tree': dt,
        'feature_columns': FEATURE_COLUMNS,
        'feature_names': FEATURE_NAMES,
        'metrics': {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'tree_rules': tree_rules,
            'importances': importances,
            'criterion': criterion,
            'max_depth': max_depth,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    }

    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)

    return artifact

def load_model(model_path='model.pkl'):
    """Loads trained model artifact from disk."""
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, 'rb') as f:
            artifact = pickle.load(f)
        return artifact
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def generate_recommendations(attendance, internal_marks, study_hours, absences, previous_grade):
    """
    Association Rule Heuristics linking student attributes to targeted interventions.
    """
    recs = []

    # Rule 1: Attendance Risk
    if attendance < 75.0:
        recs.append({
            'type': 'danger',
            'title': 'High Attendance Risk',
            'rule': 'Attendance < 75%',
            'text': f'Current attendance is {attendance}%. Mandatory attendance counseling and class catch-up sessions required.'
        })
    elif attendance >= 85.0:
        recs.append({
            'type': 'success',
            'title': 'Strong Attendance Record',
            'rule': 'Attendance ≥ 85%',
            'text': f'Excellent attendance rate at {attendance}%. Continue maintaining high classroom engagement.'
        })

    # Rule 2: Internal Performance Risk
    if internal_marks < 20:
        recs.append({
            'type': 'danger',
            'title': 'Low Internal Assessment Score',
            'rule': 'Internal Marks < 20 / 50',
            'text': f'Internal marks stand at {internal_marks}/50. Enrollment in mandatory academic remedial support & tutoring is recommended.'
        })
    elif internal_marks >= 35:
        recs.append({
            'type': 'success',
            'title': 'Solid Internal Assessment',
            'rule': 'Internal Marks ≥ 35 / 50',
            'text': f'Good internal score ({internal_marks}/50). Keep up the study momentum for final exams.'
        })

    # Rule 3: Study Time Deficit
    if study_hours < 10.0:
        recs.append({
            'type': 'warning',
            'title': 'Study Hours Deficit',
            'rule': 'Study Hours < 10 hrs/wk',
            'text': f'Weekly self-study is only {study_hours} hrs/wk. Increase structured study time to at least 12–15 hours weekly.'
        })

    # Rule 4: Excessive Unexcused Absences
    if absences > 7:
        recs.append({
            'type': 'warning',
            'title': 'High Unexcused Absences',
            'rule': 'Absences > 7 days',
            'text': f'Total unexcused absences ({absences} days) impair course learning continuity. Meet academic counselor.'
        })

    # Rule 5: Previous Grade Context
    if previous_grade in ['D', 'F']:
        recs.append({
            'type': 'info',
            'title': 'Prior Academic Weakness',
            'rule': 'Previous Grade in {D, F}',
            'text': 'History of past low grades requires early midterm progress reviews and faculty mentoring.'
        })

    if not recs:
        recs.append({
            'type': 'success',
            'title': 'Optimal Academic Profile',
            'rule': 'All Metrics Nominal',
            'text': 'Student demonstrates steady study habits, high attendance, and healthy assessment marks.'
        })

    return recs

def predict_student_outcome(input_data, model_path='model.pkl'):
    """
    Evaluates student feature payload, predicts Pass/Fail with calibrated confidence,
    generates tailored academic recommendations, and updates transaction log.
    """
    artifact = load_model(model_path)
    if artifact is None:
        return {'error': 'Trained model artifact not found. Please train model first.'}

    calibrated_model = artifact['model']

    # Extract input fields
    gender = input_data.get('gender', 'Male')
    age = int(input_data.get('age', 20))
    attendance = float(input_data.get('attendance', 80.0))
    study_hours = float(input_data.get('study_hours', 10.0))
    internal_marks = float(input_data.get('internal_marks', 30.0))
    previous_grade = input_data.get('previous_grade', 'B')
    absences = int(input_data.get('absences', 3))

    # Ordinal & Categorical Encodings
    grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    gender_map = {'Male': 0, 'Female': 1}

    grade_encoded = grade_map.get(previous_grade, 2)
    gender_encoded = gender_map.get(gender, 0)

    # Feature vector matching FEATURE_COLUMNS order
    # ['Gender_Encoded', 'Age', 'Attendance', 'Study_Hours', 'Internal_Marks', 'Grade_Encoded', 'Absences']
    feature_vector = np.array([[
        gender_encoded,
        age,
        attendance,
        study_hours,
        internal_marks,
        grade_encoded,
        absences
    ]])

    # Inference
    pred_class = calibrated_model.predict(feature_vector)[0]
    probabilities = calibrated_model.predict_proba(feature_vector)[0]

    fail_prob = round(probabilities[0] * 100, 1)
    pass_prob = round(probabilities[1] * 100, 1)

    outcome = "Pass" if pred_class == 1 else "Fail"
    confidence = pass_prob if outcome == "Pass" else fail_prob

    # Generate Heuristic Recommendations
    recommendations = generate_recommendations(attendance, internal_marks, study_hours, absences, previous_grade)

    result_dict = {
        'id': f"PRED-{int(datetime.now().timestamp())}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'inputs': {
            'gender': gender,
            'age': age,
            'attendance': attendance,
            'study_hours': study_hours,
            'internal_marks': internal_marks,
            'previous_grade': previous_grade,
            'absences': absences
        },
        'outcome': outcome,
        'confidence': confidence,
        'pass_probability': pass_prob,
        'fail_probability': fail_prob,
        'recommendations': recommendations
    }

    # Save to prediction transaction log
    save_prediction_history(result_dict)

    return result_dict

def save_prediction_history(entry, history_path='prediction_history.json'):
    """Appends prediction record to persistent transaction log."""
    history = get_prediction_history(history_path)
    history.insert(0, entry)  # prepend latest

    # Cap log at 100 entries
    history = history[:100]

    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error writing prediction history: {e}")

def get_prediction_history(history_path='prediction_history.json'):
    """Reads transaction log from disk."""
    if not os.path.exists(history_path):
        return []
    try:
        with open(history_path, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def clear_prediction_history(history_path='prediction_history.json'):
    """Resets transaction log."""
    try:
        with open(history_path, 'w') as f:
            json.dump([], f)
        return True
    except Exception as e:
        print(f"Error clearing history: {e}")
        return False
