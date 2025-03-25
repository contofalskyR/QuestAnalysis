#!/usr/bin/env python
"""
Enhanced cluster analysis to visualize both pre and post training thresholds
for Cluster 1 vs other experts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats

# Define the list of learner participant IDs (experts)
learner_ids = [
    '053539_Utility_discrimination_2024_2cat_2025-03-17_00h40.52.420',
    '059774_Utility_discrimination_2024_2cat_2025-03-18_22h04.37.229',
    '171876_Utility_discrimination_2024_2cat_2025-03-15_10h05.58.976',
    '215334_Utility_discrimination_2024_2cat_2025-03-15_20h35.35.895',
    '229150_Utility_discrimination_2024_2cat_2025-03-14_20h34.33.419',
    '249109_Utility_discrimination_2024_2cat_2025-03-17_10h37.42.839',
    '485127_Utility_discrimination_2024_2cat_2025-03-17_23h23.40.430',
    '501271_Utility_discrimination_2024_2cat_2025-03-14_15h23.42.123',
    '509243_Utility_discrimination_2024_2cat_2025-03-14_00h08.13.599',
    '516600_Utility_discrimination_2024_2cat_2025-03-12_14h14.48.442',
    '537791_Utility_discrimination_2024_2cat_2025-03-14_20h45.35.085',
    '649603_Utility_discrimination_2024_2cat_2025-03-15_16h48.20.064',
    '664805_Utility_discrimination_2024_2cat_2025-03-14_19h38.57.391',
    '699892_Utility_discrimination_2024_2cat_2025-03-15_14h12.47.943',
    '775537_Utility_discrimination_2024_2cat_2025-03-17_14h12.46.286',
    '788879_Utility_discrimination_2024_2cat_2025-03-15_15h55.58.904',
    '838091_Utility_discrimination_2024_2cat_2025-03-13_21h26.42.203',
    '856852_Utility_discrimination_2024_2cat_2025-03-14_11h38.16.574',
    '953060_Utility_discrimination_2024_2cat_2025-03-17_16h04.09.188',
    'Nidhi_Shah_Utility_discrimination_2024_2cat_2025-03-19_21h14.14.065'
]

# Define alpha angles
alpha_angles = np.array([0, 22.5, 45, 67.5, 90])

def analyze_expert_clusters(csv_file, save_dir='./results', delta_col='delta_threshold_mean', 
                          pre_col='pre_threshold_mean', post_col='post_threshold_mean'):
    """
    Analyze expert clusters and compare pre and post training thresholds.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with threshold data
    save_dir : str
        Directory to save output figures
    delta_col : str
        Column name for delta thresholds
    pre_col : str
        Column name for pre-training thresholds
    post_col : str
        Column name for post-training thresholds
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")
    
    # Mark experts
    df['is_expert'] = df['participant'].isin(learner_ids)
    expert_df = df[df['is_expert']]
    
    # Check if we have enough experts
    num_experts = expert_df['participant'].nunique()
    print(f"Found {num_experts} experts in the data")
    
    if num_experts < 3:
        print("Not enough experts for clustering analysis.")
        return
    
    # Create participant-level feature matrix
    expert_profiles = []
    
    for participant in expert_df['participant'].unique():
        participant_data = expert_df[expert_df['participant'] == participant]
        
        # Skip participants without complete data
        if len(participant_data) < len(alpha_angles):
            print(f"Skipping {participant}: incomplete data")
            continue
            
        profile = {
            'participant': participant,
            'delta_thresholds': [],  
            'pre_thresholds': [],
            'post_thresholds': []
        }
        
        # Gather data for each feature index
        for feature_idx in range(len(alpha_angles)):
            feature_data = participant_data[participant_data['feature_index'] == feature_idx]
            
            if not feature_data.empty:
                profile['delta_thresholds'].append(feature_data[delta_col].values[0])
                profile['pre_thresholds'].append(feature_data[pre_col].values[0])
                profile['post_thresholds'].append(feature_data[post_col].values[0])
            else:
                profile['delta_thresholds'].append(np.nan)
                profile['pre_thresholds'].append(np.nan)
                profile['post_thresholds'].append(np.nan)
        
        # Only include participants with complete data
        if not any(np.isnan(profile['delta_thresholds'])):
            expert_profiles.append(profile)
    
    # Check if we have enough profiles
    if len(expert_profiles) < 3:
        print(f"Only {len(expert_profiles)} experts have complete data across all features.")
        return
    
    # Create feature matrix for clustering
    X_delta = np.array([profile['delta_thresholds'] for profile in expert_profiles])
    
    # Standardize the data for clustering
    scaler = StandardScaler()
    X_delta_scaled = scaler.fit_transform(X_delta)
    
    # Cluster using k=2 (based on previous elbow method results)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_delta_scaled)
    
    # Add cluster labels to profiles
    for i, profile in enumerate(expert_profiles):
        profile['cluster'] = int(cluster_labels[i])
    
    # Get Cluster 1 participants
    cluster1_participants = [p['participant'] for p in expert_profiles if p['cluster'] == 1]
    print(f"Cluster 1 participants: {cluster1_participants}")
    
    # Create a visualization comparing pre AND post thresholds
    plt.figure(figsize=(12, 10))
    
    # Plot pre-training thresholds
    plt.subplot(2, 1, 1)
    
    # Get data for each group
    pre_cluster1 = np.mean([p['pre_thresholds'] for p in expert_profiles if p['cluster'] == 1], axis=0)
    pre_others = np.mean([p['pre_thresholds'] for p in expert_profiles if p['cluster'] == 0], axis=0)
    
    plt.plot(alpha_angles, pre_cluster1, 'o-', color='red', linewidth=2, label='Cluster 1')
    plt.plot(alpha_angles, pre_others, 's-', color='blue', linewidth=2, label='Other Experts')
    
    plt.xlabel('Alpha Angle (°)', fontsize=12)
    plt.ylabel('Pre-Training Threshold', fontsize=12)
    plt.title('Pre-Training Thresholds: Cluster 1 vs. Others', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot post-training thresholds
    plt.subplot(2, 1, 2)
    
    # Get data for each group
    post_cluster1 = np.mean([p['post_thresholds'] for p in expert_profiles if p['cluster'] == 1], axis=0)
    post_others = np.mean([p['post_thresholds'] for p in expert_profiles if p['cluster'] == 0], axis=0)
    
    plt.plot(alpha_angles, post_cluster1, 'o-', color='red', linewidth=2, label='Cluster 1')
    plt.plot(alpha_angles, post_others, 's-', color='blue', linewidth=2, label='Other Experts')
    
    plt.xlabel('Alpha Angle (°)', fontsize=12)
    plt.ylabel('Post-Training Threshold', fontsize=12)
    plt.title('Post-Training Thresholds: Cluster 1 vs. Others', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pre_post_comparison.png'), dpi=300)
    
    # Create a figure with threshold changes across training
    plt.figure(figsize=(12, 10))
    
    # Plot the change from pre to post for both groups
    plt.subplot(2, 1, 1)
    
    # Bar chart showing pre and post for Cluster 1
    x = np.arange(len(alpha_angles))
    width = 0.35
    
    plt.bar(x - width/2, pre_cluster1, width, label='Pre-Training', color='lightcoral')
    plt.bar(x + width/2, post_cluster1, width, label='Post-Training', color='darkred')
    
    plt.xlabel('Alpha Angle (°)', fontsize=12)
    plt.ylabel('Threshold', fontsize=12)
    plt.title('Cluster 1: Pre vs. Post Thresholds', fontsize=14)
    plt.xticks(x, [f'{a}°' for a in alpha_angles])
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    # Bar chart showing pre and post for other experts
    plt.subplot(2, 1, 2)
    
    plt.bar(x - width/2, pre_others, width, label='Pre-Training', color='lightskyblue')
    plt.bar(x + width/2, post_others, width, label='Post-Training', color='darkblue')
    
    plt.xlabel('Alpha Angle (°)', fontsize=12)
    plt.ylabel('Threshold', fontsize=12)
    plt.title('Other Experts: Pre vs. Post Thresholds', fontsize=14)
    plt.xticks(x, [f'{a}°' for a in alpha_angles])
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'threshold_changes.png'), dpi=300)
    
    # Create a direct comparison of thresholds
    plt.figure(figsize=(12, 6))
    
    # Calculate the fold change in thresholds (post/pre)
    fold_change_c1 = post_cluster1 / pre_cluster1
    fold_change_others = post_others / pre_others
    
    plt.plot(alpha_angles, fold_change_c1, 'o-', color='red', linewidth=2, 
             label=f'Cluster 1 (n={len([p for p in expert_profiles if p["cluster"] == 1])})')
    plt.plot(alpha_angles, fold_change_others, 's-', color='blue', linewidth=2, 
             label=f'Other Experts (n={len([p for p in expert_profiles if p["cluster"] == 0])})')
    
    plt.xlabel('Alpha Angle (°)', fontsize=12)
    plt.ylabel('Fold Change in Threshold (Post/Pre)', fontsize=12)
    plt.title('Relative Change in Thresholds After Training', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No Change')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fold_change_comparison.png'), dpi=300)
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    
    print("\nPre-Training Thresholds:")
    print(f"Cluster 1 (n={len([p for p in expert_profiles if p['cluster'] == 1])}): {pre_cluster1}")
    print(f"Other Experts (n={len([p for p in expert_profiles if p['cluster'] == 0])}): {pre_others}")
    
    print("\nPost-Training Thresholds:")
    print(f"Cluster 1: {post_cluster1}")
    print(f"Other Experts: {post_others}")
    
    print("\nMean Threshold Change (Post - Pre):")
    print(f"Cluster 1: {post_cluster1 - pre_cluster1}")
    print(f"Other Experts: {post_others - pre_others}")
    
    print("\nFold Change in Thresholds (Post/Pre):")
    print(f"Cluster 1: {fold_change_c1}")
    print(f"Other Experts: {fold_change_others}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze expert clusters and compare pre and post training thresholds')
    parser.add_argument('--input', '-i', required=True, help='Path to CSV file with threshold data')
    parser.add_argument('--output', '-o', default='./results', help='Directory to save results')
    parser.add_argument('--delta-col', default='delta_threshold_mean', help='Column for delta thresholds')
    parser.add_argument('--pre-col', default='pre_threshold_mean', help='Column for pre-training thresholds')
    parser.add_argument('--post-col', default='post_threshold_mean', help='Column for post-training thresholds')
    
    args = parser.parse_args()
    
    analyze_expert_clusters(
        args.input,
        save_dir=args.output,
        delta_col=args.delta_col,
        pre_col=args.pre_col,
        post_col=args.post_col
    )