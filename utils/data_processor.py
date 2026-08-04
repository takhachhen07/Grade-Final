import os
import pandas as pd
import numpy as np

def get_dataset(filepath='uploaded_dataset.csv'):
    """Extract operational dataset from uploaded CSV file. Returns empty DataFrame if no file is uploaded yet."""
    if not os.path.exists(filepath):
        print(f"Dataset file {filepath} not found. Awaiting CSV file upload...")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"Dataset file {filepath} is empty.")
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"Error reading dataset from {filepath}: {e}")
        return pd.DataFrame()


def clean_dataframe(df):
    """
    ETL Transformation: Missing value imputation (mean/mode),
    categorical string normalization, and numeric boundary enforcement.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df_clean = df.copy()

    # Define feature types
    numeric_cols = ['Age', 'Attendance', 'Study_Hours', 'Internal_Marks', 'Absences']

    # Convert numeric columns safely and impute missing with mean
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            mean_val = df_clean[col].mean()
            if pd.isna(mean_val):
                mean_val = 0.0
            df_clean[col] = df_clean[col].fillna(mean_val)

    # Enforce domain boundaries
    if 'Attendance' in df_clean.columns:
        df_clean['Attendance'] = df_clean['Attendance'].clip(lower=0.0, upper=100.0).round(1)
    if 'Internal_Marks' in df_clean.columns:
        df_clean['Internal_Marks'] = df_clean['Internal_Marks'].clip(lower=0, upper=50).round(0).astype(int)
    if 'Study_Hours' in df_clean.columns:
        df_clean['Study_Hours'] = df_clean['Study_Hours'].clip(lower=0.0, upper=40.0).round(1)
    if 'Absences' in df_clean.columns:
        df_clean['Absences'] = df_clean['Absences'].clip(lower=0, upper=30).astype(int)
    if 'Age' in df_clean.columns:
        df_clean['Age'] = df_clean['Age'].clip(lower=15, upper=30).astype(int)

    # Normalize Result column to strictly 'Pass' or 'Fail'
    if 'Result' in df_clean.columns:
        def norm_res(val):
            if pd.isna(val) or val is None:
                return np.nan
            s = str(val).strip().lower()
            if s in ['pass', 'passed', 'p', '1', 'true', 'yes', 'p.a.s.s.']:
                return 'Pass'
            elif s in ['fail', 'failed', 'f', '0', 'false', 'no', 'f.a.i.l.']:
                return 'Fail'
            elif 'pass' in s:
                return 'Pass'
            elif 'fail' in s:
                return 'Fail'
            return 'Pass'

        df_clean['Result'] = df_clean['Result'].apply(norm_res)
        if df_clean['Result'].isnull().sum() > 0:
            mode_res = df_clean['Result'].mode()
            fill_res = mode_res[0] if not mode_res.empty and pd.notna(mode_res[0]) else 'Pass'
            df_clean['Result'] = df_clean['Result'].fillna(fill_res)

    # Normalize Gender
    if 'Gender' in df_clean.columns:
        def norm_gender(val):
            if pd.isna(val) or val is None:
                return np.nan
            s = str(val).strip().lower()
            if s in ['male', 'm', 'boy', 'man', '1']:
                return 'Male'
            elif s in ['female', 'f', 'girl', 'woman', '0']:
                return 'Female'
            return 'Male'

        df_clean['Gender'] = df_clean['Gender'].apply(norm_gender)
        if df_clean['Gender'].isnull().sum() > 0:
            mode_gen = df_clean['Gender'].mode()
            fill_gen = mode_gen[0] if not mode_gen.empty and pd.notna(mode_gen[0]) else 'Male'
            df_clean['Gender'] = df_clean['Gender'].fillna(fill_gen)

    # Normalize Previous_Grade
    if 'Previous_Grade' in df_clean.columns:
        def norm_grade(val):
            if pd.isna(val) or val is None:
                return np.nan
            s = str(val).strip().upper()
            if s in ['A', 'B', 'C', 'D', 'F']:
                return s
            elif s.startswith('A'):
                return 'A'
            elif s.startswith('B'):
                return 'B'
            elif s.startswith('C'):
                return 'C'
            elif s.startswith('D'):
                return 'D'
            return 'B'

        df_clean['Previous_Grade'] = df_clean['Previous_Grade'].apply(norm_grade)
        if df_clean['Previous_Grade'].isnull().sum() > 0:
            mode_grd = df_clean['Previous_Grade'].mode()
            fill_grd = mode_grd[0] if not mode_grd.empty and pd.notna(mode_grd[0]) else 'B'
            df_clean['Previous_Grade'] = df_clean['Previous_Grade'].fillna(fill_grd)

    return df_clean

def encode_dataframe(df):
    """
    ETL Encoding: Encodes ordinal and nominal categorical features into numeric formats.
    """
    df_encoded = df.copy()

    # Ordinal Encoding for Previous_Grade
    grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    if 'Previous_Grade' in df_encoded.columns:
        df_encoded['Grade_Encoded'] = df_encoded['Previous_Grade'].map(grade_map).fillna(2).astype(int)

    # Binary Encoding for Gender
    gender_map = {'Male': 0, 'Female': 1}
    if 'Gender' in df_encoded.columns:
        df_encoded['Gender_Encoded'] = df_encoded['Gender'].map(gender_map).fillna(0).astype(int)

    # Target Encoding for Result
    result_map = {'Pass': 1, 'Fail': 0}
    if 'Result' in df_encoded.columns:
        df_encoded['Result_Encoded'] = df_encoded['Result'].map(result_map).fillna(0).astype(int)

    return df_encoded

def get_summary_stats(df):
    """Generates OLAP multidimensional summary statistics safely."""
    if df is None or df.empty:
        return {
            'total_records': 0,
            'pass_count': 0,
            'fail_count': 0,
            'pass_rate': 0.0,
            'avg_attendance': 0.0,
            'avg_study_hours': 0.0,
            'avg_internal_marks': 0.0,
            'avg_absences': 0.0
        }

    df_clean = clean_dataframe(df)
    total_records = len(df_clean)

    if total_records == 0 or 'Result' not in df_clean.columns:
        return {
            'total_records': 0,
            'pass_count': 0,
            'fail_count': 0,
            'pass_rate': 0.0,
            'avg_attendance': 0.0,
            'avg_study_hours': 0.0,
            'avg_internal_marks': 0.0,
            'avg_absences': 0.0
        }

    pass_count = len(df_clean[df_clean['Result'] == 'Pass'])
    fail_count = len(df_clean[df_clean['Result'] == 'Fail'])
    pass_rate = round((pass_count / total_records) * 100, 1) if total_records > 0 else 0.0

    avg_attendance = round(df_clean['Attendance'].mean(), 1) if 'Attendance' in df_clean.columns and not df_clean['Attendance'].isnull().all() else 0.0
    avg_study_hours = round(df_clean['Study_Hours'].mean(), 1) if 'Study_Hours' in df_clean.columns and not df_clean['Study_Hours'].isnull().all() else 0.0
    avg_internal_marks = round(df_clean['Internal_Marks'].mean(), 1) if 'Internal_Marks' in df_clean.columns and not df_clean['Internal_Marks'].isnull().all() else 0.0
    avg_absences = round(df_clean['Absences'].mean(), 1) if 'Absences' in df_clean.columns and not df_clean['Absences'].isnull().all() else 0.0

    return {
        'total_records': total_records,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'pass_rate': pass_rate,
        'avg_attendance': avg_attendance,
        'avg_study_hours': avg_study_hours,
        'avg_internal_marks': avg_internal_marks,
        'avg_absences': avg_absences
    }
