import os
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np

from utils.data_processor import get_dataset, clean_dataframe, get_summary_stats
from utils.visualizer import generate_all_visualizations
from utils.olap_engine import compute_olap_cube
from utils.association_rules import run_apriori_mining
from utils.clustering import run_kmeans_clustering

app = Flask(__name__)
app.secret_key = 'edu_analytics_com470_secret_key'

DATASET_PATH = 'uploaded_dataset.csv'

def init_app():
    """Initializes dataset and pre-renders data visualizations."""
    if os.path.exists(DATASET_PATH):
        df = get_dataset(DATASET_PATH)
        if not df.empty:
            generate_all_visualizations(df, None)

@app.route('/')
def index():
    """Home Overview Dashboard."""
    df = get_dataset(DATASET_PATH)
    stats = get_summary_stats(df)
    return render_template('index.html', stats=stats)

@app.route('/dataset')
def dataset():
    """Dataset Explorer & ETL Data Cleaning View."""
    df = get_dataset(DATASET_PATH)
    stats = get_summary_stats(df)
    
    # Check for missing values in raw dataset
    raw_missing = df.isnull().sum().to_dict()
    has_missing = any(v > 0 for v in raw_missing.values())
    raw_missing_list = [{'col': k, 'count': v} for k, v in raw_missing.items() if v > 0]

    records = df.head(100).to_dict(orient='records')
    columns = df.columns.tolist()

    return render_template(
        'dataset.html',
        records=records,
        columns=columns,
        total_count=len(df),
        stats=stats,
        raw_missing=raw_missing,
        raw_missing_list=raw_missing_list,
        has_missing=has_missing
    )

@app.route('/olap')
def olap():
    """OLAP Data Cube Multidimensional Analytics View."""
    df = get_dataset(DATASET_PATH)
    group_dims = request.args.getlist('dims') or ['Previous_Grade', 'Attendance_Tier']
    slice_dim = request.args.get('slice_dim', '')
    slice_val = request.args.get('slice_val', '')
    cube_res = compute_olap_cube(df, slice_dim=slice_dim, slice_val=slice_val, group_by_dims=group_dims)
    stats = get_summary_stats(df)
    return render_template('olap.html', cube=cube_res, stats=stats, selected_dims=group_dims, slice_dim=slice_dim, slice_val=slice_val)

@app.route('/association-rules')
def association_rules():
    """Association Rule Mining (Apriori Algorithm) View."""
    df = get_dataset(DATASET_PATH)
    min_sup = float(request.args.get('min_support', 0.1) or 0.1)
    min_conf = float(request.args.get('min_confidence', 0.4) or 0.4)
    mining_res = run_apriori_mining(df, min_support=min_sup, min_confidence=min_conf)
    return render_template('association_rules.html', mining=mining_res, min_support=min_sup, min_confidence=min_conf)

@app.route('/clustering')
def clustering():
    """K-Means Cluster Analysis View."""
    df = get_dataset(DATASET_PATH)
    k = int(request.args.get('k', 3) or 3)
    cluster_res = run_kmeans_clustering(df, n_clusters=k)
    return render_template('clustering.html', clustering=cluster_res, k=k)

# --- REST API Endpoints ---

@app.route('/api/upload-csv', methods=['POST'])
def api_upload_csv():
    """Stages new CSV records into Operational Data Store and cleans missing values via ETL."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'message': 'Only CSV files are allowed'}), 400

    try:
        df_uploaded = pd.read_csv(file)
        if df_uploaded.empty:
            return jsonify({'success': False, 'message': 'Uploaded CSV file is empty'}), 400

        # Normalize column headers: strip whitespace
        df_uploaded.columns = df_uploaded.columns.str.strip()

        # Flexible column mapping for case variations / common aliases
        column_mapping = {}
        for col in df_uploaded.columns:
            c_lower = col.lower().replace(' ', '_').replace('-', '_')
            if c_lower in ['gender', 'sex']:
                column_mapping[col] = 'Gender'
            elif c_lower in ['age']:
                column_mapping[col] = 'Age'
            elif c_lower in ['attendance', 'attendance_pct', 'attendance_%', 'attendance_rate']:
                column_mapping[col] = 'Attendance'
            elif c_lower in ['study_hours', 'studyhours', 'study_time', 'weekly_study_hours']:
                column_mapping[col] = 'Study_Hours'
            elif c_lower in ['internal_marks', 'internal_score', 'marks', 'internals', 'test_score']:
                column_mapping[col] = 'Internal_Marks'
            elif c_lower in ['previous_grade', 'prev_grade', 'grade', 'past_grade']:
                column_mapping[col] = 'Previous_Grade'
            elif c_lower in ['absences', 'absent', 'absent_days']:
                column_mapping[col] = 'Absences'
            elif c_lower in ['result', 'outcome', 'status', 'passed', 'pass_fail']:
                column_mapping[col] = 'Result'

        if column_mapping:
            df_uploaded.rename(columns=column_mapping, inplace=True)

        # Ensure required columns exist
        required_cols = ['Gender', 'Age', 'Attendance', 'Study_Hours', 'Internal_Marks', 'Previous_Grade', 'Absences', 'Result']
        for col in required_cols:
            if col not in df_uploaded.columns:
                df_uploaded[col] = np.nan

        # Save uploaded dataset into ODS storage
        df_uploaded.to_csv(DATASET_PATH, index=False)

        # Execute ETL cleaning
        df_cleaned = clean_dataframe(df_uploaded)
        generate_all_visualizations(df_cleaned, None)

        return jsonify({
            'success': True,
            'message': f'CSV uploaded and staged into Operational Data Store ({len(df_uploaded)} records)!'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing CSV: {str(e)}'}), 500

@app.route('/api/clean-data', methods=['POST'])
def api_clean_data():
    """Executes missing value cleaning on Operational Data Store dataset."""
    try:
        df = get_dataset(DATASET_PATH)
        df_cleaned = clean_dataframe(df)
        df_cleaned.to_csv(DATASET_PATH, index=False)
        generate_all_visualizations(df_cleaned, None)

        stats = get_summary_stats(df_cleaned)

        return jsonify({
            'success': True,
            'message': 'ETL Data cleaning executed successfully! Missing values resolved.',
            'stats': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Data cleaning error: {str(e)}'}), 500

# Initialize app dependencies on import/run
init_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
