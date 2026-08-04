import pandas as pd
import numpy as np

def compute_olap_cube(df, slice_dim=None, slice_val=None, dice_filters=None, group_by_dims=None):
    """
    Computes an OLAP Data Cube with Slice, Dice, Roll-Up, and Drill-Down capabilities.
    
    Measures evaluated:
    - Record Count
    - Pass Count & Pass Rate (%)
    - Mean Attendance (%)
    - Mean Study Hours (hrs/wk)
    - Mean Internal Marks (/50)
    - Mean Absences (days)
    """
    if df is None or df.empty:
        return {'cube_data': [], 'dimensions': [], 'summary': {}}

    cube_df = df.copy()

    # Create derived dimension bins for OLAP slicing and dicing
    if 'Attendance' in cube_df.columns:
        cube_df['Attendance_Tier'] = pd.cut(
            cube_df['Attendance'],
            bins=[-1, 70, 85, 101],
            labels=['Low (<70%)', 'Moderate (70-85%)', 'High (>85%)']
        ).astype(str)

    if 'Study_Hours' in cube_df.columns:
        cube_df['Study_Tier'] = pd.cut(
            cube_df['Study_Hours'],
            bins=[-1, 8, 15, 100],
            labels=['Low (<8h)', 'Moderate (8-15h)', 'High (>15h)']
        ).astype(str)

    if 'Internal_Marks' in cube_df.columns:
        cube_df['Internal_Tier'] = pd.cut(
            cube_df['Internal_Marks'],
            bins=[-1, 20, 35, 51],
            labels=['Low (<20)', 'Medium (20-35)', 'High (>35)']
        ).astype(str)

    # 1. Apply Slice (single dimension filter)
    if slice_dim and slice_val and slice_dim in cube_df.columns:
        cube_df = cube_df[cube_df[slice_dim].astype(str) == str(slice_val)]

    # 2. Apply Dice (multi-dimensional sub-cube filter)
    if dice_filters and isinstance(dice_filters, dict):
        for dim, val in dice_filters.items():
            if val and dim in cube_df.columns:
                cube_df = cube_df[cube_df[dim].astype(str) == str(val)]

    # Default grouping dimensions if none provided
    if not group_by_dims:
        group_by_dims = ['Previous_Grade', 'Attendance_Tier']

    # Filter out missing columns
    group_by_dims = [d for d in group_by_dims if d in cube_df.columns]

    if not group_by_dims:
        group_by_dims = ['Gender']

    # 3. Compute Aggregations across dimensions (Roll-Up / Drill-Down)
    grouped_rows = []
    if not cube_df.empty:
        grouped = cube_df.groupby(group_by_dims, observed=False)
        for name, group in grouped:
            if group.empty:
                continue
            
            dim_key = name if isinstance(name, tuple) else (name,)
            dim_dict = {group_by_dims[i]: dim_key[i] for i in range(len(group_by_dims))}
            
            total = len(group)
            pass_cnt = len(group[group['Result'] == 'Pass']) if 'Result' in group.columns else 0
            fail_cnt = len(group[group['Result'] == 'Fail']) if 'Result' in group.columns else 0
            pass_rate = round((pass_cnt / total) * 100, 1) if total > 0 else 0.0
            
            avg_att = round(group['Attendance'].mean(), 1) if 'Attendance' in group.columns else 0.0
            avg_std = round(group['Study_Hours'].mean(), 1) if 'Study_Hours' in group.columns else 0.0
            avg_marks = round(group['Internal_Marks'].mean(), 1) if 'Internal_Marks' in group.columns else 0.0
            avg_abs = round(group['Absences'].mean(), 1) if 'Absences' in group.columns else 0.0

            entry = {
                **dim_dict,
                'count': total,
                'pass_count': pass_cnt,
                'fail_count': fail_cnt,
                'pass_rate': pass_rate,
                'avg_attendance': avg_att,
                'avg_study_hours': avg_std,
                'avg_internal_marks': avg_marks,
                'avg_absences': avg_abs
            }
            grouped_rows.append(entry)

    # Sort grouped rows by count descending
    grouped_rows.sort(key=lambda x: x['count'], reverse=True)

    summary = {
        'total_subcube_records': len(cube_df),
        'overall_pass_rate': round((len(cube_df[cube_df['Result'] == 'Pass']) / len(cube_df) * 100), 1) if not cube_df.empty and 'Result' in cube_df.columns else 0.0,
        'group_by_dims': group_by_dims
    }

    return {
        'cube_data': grouped_rows,
        'dimensions': group_by_dims,
        'summary': summary
    }
