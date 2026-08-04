import pandas as pd
import numpy as np

def run_kmeans_clustering(df, n_clusters=3):
    """
    K-Means Clustering implementation for Student Segment Analysis.
    
    Features normalized for clustering:
    - Attendance (%)
    - Study_Hours (hrs/wk)
    - Internal_Marks (/50)
    - Absences (days)
    """
    if df is None or df.empty:
        return {'clusters': [], 'centroids': [], 'stats': {}}

    features = ['Attendance', 'Study_Hours', 'Internal_Marks', 'Absences']
    
    # Check if features exist
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2:
        return {'clusters': [], 'centroids': [], 'stats': {}}

    # Clean subset
    cluster_df = df[available_features].dropna().copy()
    if len(cluster_df) < n_clusters:
        return {'clusters': [], 'centroids': [], 'stats': {}}

    # Standardize features for distance calculation
    means = cluster_df[available_features].mean()
    stds = cluster_df[available_features].std().replace(0, 1)
    
    scaled_data = (cluster_df[available_features] - means) / stds

    # Custom K-Means iteration
    np.random.seed(42)
    sample_indices = np.random.choice(len(scaled_data), size=n_clusters, replace=False)
    centroids = scaled_data.iloc[sample_indices].values.copy()

    labels = np.zeros(len(scaled_data), dtype=int)
    for iteration in range(20):
        # Assign cluster label by Euclidean distance
        distances = np.sqrt(((scaled_data.values[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(labels, new_labels) and iteration > 0:
            break
        labels = new_labels

        # Update centroids
        for c in range(n_clusters):
            cluster_members = scaled_data.values[labels == c]
            if len(cluster_members) > 0:
                centroids[c] = cluster_members.mean(axis=0)

    # Map back unscaled centroids
    unscaled_centroids = (centroids * stds.values) + means.values

    cluster_df['Cluster_ID'] = labels

    # Build cluster profile analysis
    cluster_profiles = []
    for c in range(n_clusters):
        c_members = cluster_df[cluster_df['Cluster_ID'] == c]
        count = len(c_members)
        
        avg_att = round(c_members['Attendance'].mean(), 1) if 'Attendance' in c_members else 0.0
        avg_sh = round(c_members['Study_Hours'].mean(), 1) if 'Study_Hours' in c_members else 0.0
        avg_im = round(c_members['Internal_Marks'].mean(), 1) if 'Internal_Marks' in c_members else 0.0
        avg_abs = round(c_members['Absences'].mean(), 1) if 'Absences' in c_members else 0.0

        # Heuristic profile name
        if avg_att >= 80 and avg_im >= 30:
            profile_name = "High-Performing Achievers"
            tag_class = "badge-pass"
        elif avg_att < 70 or avg_im < 22:
            profile_name = "At-Risk Academic Warning"
            tag_class = "badge-fail"
        elif avg_sh >= 12:
            profile_name = "Dedicated High-Effort"
            tag_class = "badge-info"
        else:
            profile_name = "Moderate Baseline Performers"
            tag_class = "badge-info"

        cluster_profiles.append({
            'cluster_id': c,
            'cluster_name': f"Cluster {c + 1}: {profile_name}",
            'tag_class': tag_class,
            'count': count,
            'pct_of_total': round((count / len(cluster_df)) * 100, 1),
            'avg_attendance': avg_att,
            'avg_study_hours': avg_sh,
            'avg_internal_marks': avg_im,
            'avg_absences': avg_abs
        })

    # Prepare 2D plot points (x: Attendance, y: Internal Marks)
    points = []
    for idx, row in cluster_df.iterrows():
        points.append({
            'x': round(row['Attendance'], 1),
            'y': round(row['Internal_Marks'], 1),
            'sh': round(row['Study_Hours'], 1),
            'cluster': int(row['Cluster_ID'])
        })

    return {
        'cluster_profiles': cluster_profiles,
        'points': points[:200],  # sample 200 for frontend rendering
        'stats': {
            'total_clustered_records': len(cluster_df),
            'n_clusters': n_clusters,
            'features_used': available_features
        }
    }
