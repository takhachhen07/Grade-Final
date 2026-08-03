import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from utils.data_processor import clean_dataframe

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
BG_COLOR = '#f8fafc'
PRIMARY_COLOR = '#2563eb'
SECONDARY_COLOR = '#10b981'
DANGER_COLOR = '#ef4444'
TEXT_COLOR = '#1e293b'

def ensure_images_dir(images_dir='static/images'):
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

def get_result_palette(df_clean):
    """Dynamically builds a palette dictionary for any unique values in Result."""
    palette = {}
    if 'Result' in df_clean.columns:
        for val in df_clean['Result'].unique():
            if str(val).strip().lower() in ['pass', 'passed', 'p', '1']:
                palette[val] = SECONDARY_COLOR
            else:
                palette[val] = DANGER_COLOR
    if not palette:
        palette = {'Pass': SECONDARY_COLOR, 'Fail': DANGER_COLOR}
    return palette

def generate_attendance_dist(df, output_path='static/images/attendance_dist.png'):
    """Generates attendance % distribution histogram grouped by Result."""
    ensure_images_dir()
    df_clean = clean_dataframe(df)

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    palette = get_result_palette(df_clean)
    sns.histplot(
        data=df_clean,
        x='Attendance',
        hue='Result',
        palette=palette,
        kde=True,
        ax=ax,
        bins=15,
        alpha=0.6
    )

    ax.set_title('Attendance % Distribution by Student Result', fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=12)
    ax.set_xlabel('Attendance Rate (%)', fontsize=10, fontweight='bold', color=TEXT_COLOR)
    ax.set_ylabel('Student Count', fontsize=10, fontweight='bold', color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_grade_dist(df, output_path='static/images/grade_dist.png'):
    """Generates Previous Grade breakdown by Result."""
    ensure_images_dir()
    df_clean = clean_dataframe(df)

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    order = ['A', 'B', 'C', 'D', 'F']
    palette = get_result_palette(df_clean)
    sns.countplot(
        data=df_clean,
        x='Previous_Grade',
        hue='Result',
        order=order,
        palette=palette,
        ax=ax
    )

    ax.set_title('Previous Grade Distribution vs. Outcome', fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=12)
    ax.set_xlabel('Previous Letter Grade', fontsize=10, fontweight='bold', color=TEXT_COLOR)
    ax.set_ylabel('Student Count', fontsize=10, fontweight='bold', color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_pass_fail_dist(df, output_path='static/images/pass_fail_dist.png'):
    """Generates Donut chart of overall Pass vs Fail proportions."""
    ensure_images_dir()
    df_clean = clean_dataframe(df)

    counts = df_clean['Result'].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    colors = [SECONDARY_COLOR if l == 'Pass' else DANGER_COLOR for l in labels]
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
        textprops=dict(color=TEXT_COLOR, fontweight='bold')
    )

    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.set_title('ODS Dataset Pass / Fail Proportion', fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_confusion_matrix_plot(cm_dict, output_path='static/images/confusion_matrix.png'):
    """Generates Confusion Matrix heatmap plot."""
    ensure_images_dir()

    tn = cm_dict.get('tn', 0)
    fp = cm_dict.get('fp', 0)
    fn = cm_dict.get('fn', 0)
    tp = cm_dict.get('tp', 0)

    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        ax=ax,
        xticklabels=['Predicted Fail', 'Predicted Pass'],
        yticklabels=['Actual Fail', 'Actual Pass'],
        annot_kws={'size': 14, 'weight': 'bold'}
    )

    ax.set_title('Test Set Confusion Matrix', fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_feature_importance_plot(importances, output_path='static/images/feature_importance.png'):
    """Generates horizontal bar chart of Decision Tree Feature Importances."""
    ensure_images_dir()

    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=False)
    features = [x[0] for x in sorted_importances]
    values = [x[1] for x in sorted_importances]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bars = ax.barh(features, values, color=PRIMARY_COLOR, alpha=0.85, height=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                va='center', ha='left', fontsize=9, fontweight='bold', color=TEXT_COLOR)

    ax.set_xlim(0, max(values) * 1.2 if values else 1.0)
    ax.set_title('Information Gain / Gini Feature Importance', fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=12)
    ax.set_xlabel('Relative Importance Score', fontsize=10, fontweight='bold', color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_all_visualizations(df, artifact):
    """Executes full chart rendering suite."""
    ensure_images_dir()
    generate_attendance_dist(df)
    generate_grade_dist(df)
    generate_pass_fail_dist(df)

    if artifact and 'metrics' in artifact:
        metrics = artifact['metrics']
        generate_confusion_matrix_plot(metrics.get('confusion_matrix', {}))
        generate_feature_importance_plot(metrics.get('importances', {}))
