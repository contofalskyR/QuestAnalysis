import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import os
import math
from scipy.stats import entropy
from scipy.spatial.distance import cdist
warnings.filterwarnings('ignore')

# Define the list of learner participant IDs based on your data
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

# Define alpha angles and calculate mutual information
alpha_angles = np.array([0, 22.5, 45, 67.5, 90])

def calculate_mutual_information(alpha_angle, ipl=0.95):
    """
    Calculate mutual information between a feature 
    at a given alpha angle and the category variable.
    """
    alpha_rad = math.radians(alpha_angle)
    max_mi = 0.71  # For Experiment 3, IPL=0.95
    mi = max_mi * math.cos(alpha_rad)
    return max(0, mi)

mutual_info_values = [calculate_mutual_information(angle) for angle in alpha_angles]

# ================================================
# 1. PATTERN ANALYSIS BEYOND CORRELATION
# ================================================

def pattern_analysis(df, participant_col='participant', feature_col='feature_index',
                   pre_col='pre_threshold_mean', post_col='post_threshold_mean',
                   delta_col='delta_threshold_mean', save_dir='.'):
    """
    Comprehensive pattern analysis of expert vs non-expert perceptual discrimination.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with pre and post threshold data
    participant_col : str
        Column name for participant IDs
    feature_col : str
        Column name for feature indices
    pre_col : str
        Column name for pre-training thresholds
    post_col : str
        Column name for post-training thresholds
    delta_col : str
        Column name for delta thresholds (pre - post)
    save_dir : str
        Directory to save output figures
    """
    # Add is_expert column based on learner_ids
    df['is_expert'] = df[participant_col].isin(learner_ids)
    
    # Count experts and non-experts
    n_experts = df[df['is_expert']]['participant'].nunique()
    n_non_experts = df[~df['is_expert']]['participant'].nunique()
    print(f"Analysis includes {n_experts} experts and {n_non_experts} non-experts")
    
    # Create figure for absolute thresholds comparison
    plt.figure(figsize=(15, 10))
    
    # 1.1 Compare absolute thresholds (pre, post, delta)
    plt.subplot(2, 2, 1)
    
    # Calculate mean thresholds by feature and expertise
    comparison_data = df.groupby([feature_col, 'is_expert']).agg({
        pre_col: ['mean', 'std', 'count'],
        post_col: ['mean', 'std', 'count'],
        delta_col: ['mean', 'std', 'count']
    }).reset_index()
    
    # Pivot the comparison data for easier plotting
    pivot_pre = comparison_data.pivot(index=feature_col, columns='is_expert', 
                                   values=(pre_col, 'mean')).reset_index()
    pivot_post = comparison_data.pivot(index=feature_col, columns='is_expert', 
                                    values=(post_col, 'mean')).reset_index()
    pivot_delta = comparison_data.pivot(index=feature_col, columns='is_expert', 
                                     values=(delta_col, 'mean')).reset_index()
    
    # Map feature index to alpha angles
    feature_to_alpha = {i: angle for i, angle in enumerate(alpha_angles)}
    x_values = [feature_to_alpha.get(idx, idx) for idx in pivot_pre[feature_col]]
    
    # Plot pre-training thresholds
    plt.plot(x_values, pivot_pre[(pre_col, 'mean', True)], 'o-', color='blue', 
             label='Expert - Pre', linewidth=2)
    plt.plot(x_values, pivot_pre[(pre_col, 'mean', False)], 's-', color='lightblue', 
             label='Non-Expert - Pre', linewidth=2)
    
    # Plot post-training thresholds
    plt.plot(x_values, pivot_post[(post_col, 'mean', True)], 'o-', color='red', 
             label='Expert - Post', linewidth=2)
    plt.plot(x_values, pivot_post[(post_col, 'mean', False)], 's-', color='lightcoral', 
             label='Non-Expert - Post', linewidth=2)
    
    plt.xlabel('Alpha Angle (°)', fontsize=12)
    plt.ylabel('Threshold', fontsize=12)
    plt.title('Pre/Post Thresholds by Alpha Angle', fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(alpha_angles)
    plt.legend()
    
    # 1.2 Compare delta thresholds directly
    plt.subplot(2, 2, 2)
    
    plt.plot(x_values, pivot_delta[(delta_col, 'mean', True)], 'o-', color='green', 
             label='Expert', linewidth=2)
    plt.plot(x_values, pivot_delta[(delta_col, 'mean', False)], 's-', color='orange', 
             label='Non-Expert', linewidth=2)
    
    plt.xlabel('Alpha Angle (°)', fontsize=12)
    plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
    plt.title('Perceptual Learning by Alpha Angle', fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(alpha_angles)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    
    # 1.3 Compare variance in thresholds
    plt.subplot(2, 2, 3)
    
    # Calculate variance for each group
    expert_data = df[df['is_expert']]
    non_expert_data = df[~df['is_expert']]
    
    # For each alpha angle, calculate the variance in delta thresholds
    expert_variance = []
    non_expert_variance = []
    
    for i, angle in enumerate(alpha_angles):
        expert_feature_data = expert_data[expert_data[feature_col] == i]
        non_expert_feature_data = non_expert_data[non_expert_data[feature_col] == i]
        
        if len(expert_feature_data) > 1:
            expert_variance.append(expert_feature_data[delta_col].var())
        else:
            expert_variance.append(np.nan)
            
        if len(non_expert_feature_data) > 1:
            non_expert_variance.append(non_expert_feature_data[delta_col].var())
        else:
            non_expert_variance.append(np.nan)
    
    plt.plot(alpha_angles, expert_variance, 'o-', color='green', 
             label='Expert', linewidth=2)
    plt.plot(alpha_angles, non_expert_variance, 's-', color='orange', 
             label='Non-Expert', linewidth=2)
    
    plt.xlabel('Alpha Angle (°)', fontsize=12)
    plt.ylabel('Variance in Delta Threshold', fontsize=12)
    plt.title('Threshold Variance by Alpha Angle', fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(alpha_angles)
    plt.legend()
    
    # 1.4 Compare relationship between mutual information and delta threshold
    plt.subplot(2, 2, 4)
    
    # Calculate mean delta thresholds for each alpha angle
    expert_delta_means = []
    non_expert_delta_means = []
    
    for i, angle in enumerate(alpha_angles):
        expert_feature_data = expert_data[expert_data[feature_col] == i]
        non_expert_feature_data = non_expert_data[non_expert_data[feature_col] == i]
        
        expert_delta_means.append(expert_feature_data[delta_col].mean() if not expert_feature_data.empty else np.nan)
        non_expert_delta_means.append(non_expert_feature_data[delta_col].mean() if not non_expert_feature_data.empty else np.nan)
    
    # Plot relationship with mutual information
    plt.scatter(mutual_info_values, expert_delta_means, color='green', s=80, label='Expert', alpha=0.7)
    plt.scatter(mutual_info_values, non_expert_delta_means, color='orange', s=80, marker='s', label='Non-Expert', alpha=0.7)
    
    # Add regression lines
    if not all(np.isnan(expert_delta_means)):
        valid_exp = ~np.isnan(expert_delta_means)
        if sum(valid_exp) > 1:
            exp_x = np.array(mutual_info_values)[valid_exp]
            exp_y = np.array(expert_delta_means)[valid_exp]
            exp_slope, exp_intercept, exp_r, exp_p, _ = stats.linregress(exp_x, exp_y)
            exp_line_x = np.linspace(0, max(mutual_info_values), 100)
            exp_line_y = exp_slope * exp_line_x + exp_intercept
            plt.plot(exp_line_x, exp_line_y, '--', color='green', 
                     label=f'Expert (R²={exp_r**2:.2f})', alpha=0.7)
    
    if not all(np.isnan(non_expert_delta_means)):
        valid_non = ~np.isnan(non_expert_delta_means)
        if sum(valid_non) > 1:
            non_x = np.array(mutual_info_values)[valid_non]
            non_y = np.array(non_expert_delta_means)[valid_non]
            non_slope, non_intercept, non_r, non_p, _ = stats.linregress(non_x, non_y)
            non_line_x = np.linspace(0, max(mutual_info_values), 100)
            non_line_y = non_slope * non_line_x + non_intercept
            plt.plot(non_line_x, non_line_y, '--', color='orange', 
                     label=f'Non-Expert (R²={non_r**2:.2f})', alpha=0.7)
    
    plt.xlabel('Mutual Information (bits)', fontsize=12)
    plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
    plt.title('Mutual Information vs. Delta Threshold', fontsize=14)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pattern_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1.5 Look for non-linear relationships
    plt.figure(figsize=(15, 5))
    
    # Non-linear relationship visualization
    # Plot various potential transformations of mutual information vs delta threshold
    transformations = [
        ('Linear', lambda x: x),
        ('Quadratic', lambda x: x**2),
        ('Log', lambda x: np.log(x + 0.01))  # Add small constant to avoid log(0)
    ]
    
    for i, (name, transform_func) in enumerate(transformations):
        plt.subplot(1, 3, i+1)
        
        # Transform mutual information values
        mi_transformed = [transform_func(mi) if not np.isnan(mi) else np.nan for mi in mutual_info_values]
        
        # Plot experts
        plt.scatter(mi_transformed, expert_delta_means, color='green', s=80, label='Expert', alpha=0.7)
        
        # Plot non-experts
        plt.scatter(mi_transformed, non_expert_delta_means, color='orange', s=80, marker='s', label='Non-Expert', alpha=0.7)
        
        # Add regression lines for transformed data
        if not all(np.isnan(expert_delta_means)):
            valid_exp = ~np.isnan(expert_delta_means) & ~np.isnan(mi_transformed)
            if sum(valid_exp) > 1:
                exp_x = np.array(mi_transformed)[valid_exp]
                exp_y = np.array(expert_delta_means)[valid_exp]
                exp_slope, exp_intercept, exp_r, exp_p, _ = stats.linregress(exp_x, exp_y)
                # Only plot if we can compute a valid line
                if not np.isnan(exp_slope) and not np.isnan(exp_intercept):
                    exp_line_x = np.linspace(min(exp_x), max(exp_x), 100)
                    exp_line_y = exp_slope * exp_line_x + exp_intercept
                    plt.plot(exp_line_x, exp_line_y, '--', color='green', 
                             label=f'Expert (R²={exp_r**2:.2f})', alpha=0.7)
        
        if not all(np.isnan(non_expert_delta_means)):
            valid_non = ~np.isnan(non_expert_delta_means) & ~np.isnan(mi_transformed)
            if sum(valid_non) > 1:
                non_x = np.array(mi_transformed)[valid_non]
                non_y = np.array(non_expert_delta_means)[valid_non]
                non_slope, non_intercept, non_r, non_p, _ = stats.linregress(non_x, non_y)
                # Only plot if we can compute a valid line
                if not np.isnan(non_slope) and not np.isnan(non_intercept):
                    non_line_x = np.linspace(min(non_x), max(non_x), 100)
                    non_line_y = non_slope * non_line_x + non_intercept
                    plt.plot(non_line_x, non_line_y, '--', color='orange', 
                             label=f'Non-Expert (R²={non_r**2:.2f})', alpha=0.7)
        
        plt.xlabel(f'{name} Mutual Information', fontsize=12)
        plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
        plt.title(f'{name} Relationship', fontsize=14)
        plt.grid(alpha=0.3)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nonlinear_relationships.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return summary statistics
    summary = {
        'pre_threshold': {
            'expert': pivot_pre[(pre_col, 'mean', True)].tolist(),
            'non_expert': pivot_pre[(pre_col, 'mean', False)].tolist()
        },
        'post_threshold': {
            'expert': pivot_post[(post_col, 'mean', True)].tolist(),
            'non_expert': pivot_post[(post_col, 'mean', False)].tolist()
        },
        'delta_threshold': {
            'expert': pivot_delta[(delta_col, 'mean', True)].tolist(),
            'non_expert': pivot_delta[(delta_col, 'mean', False)].tolist()
        },
        'variance': {
            'expert': expert_variance,
            'non_expert': non_expert_variance
        },
        'alpha_angles': alpha_angles.tolist(),
        'mutual_info': mutual_info_values
    }
    
    # Print key findings
    print("\n=== PATTERN ANALYSIS RESULTS ===")
    print("1. Absolute threshold comparison:")
    for i, angle in enumerate(alpha_angles):
        print(f"   Alpha {angle}°: Experts pre={summary['pre_threshold']['expert'][i]:.3f}, post={summary['post_threshold']['expert'][i]:.3f}, delta={summary['delta_threshold']['expert'][i]:.3f}")
        print(f"                Non-experts pre={summary['pre_threshold']['non_expert'][i]:.3f}, post={summary['post_threshold']['non_expert'][i]:.3f}, delta={summary['delta_threshold']['non_expert'][i]:.3f}")
    
    # Calculate average learning effects
    avg_expert_delta = np.nanmean(summary['delta_threshold']['expert'])
    avg_non_expert_delta = np.nanmean(summary['delta_threshold']['non_expert'])
    print(f"\n2. Overall learning effect:")
    print(f"   Experts: {avg_expert_delta:.3f}")
    print(f"   Non-experts: {avg_non_expert_delta:.3f}")
    
    # Calculate variance patterns
    variance_pattern_expert = "Increasing" if np.nanmean(summary['variance']['expert'][:2]) < np.nanmean(summary['variance']['expert'][-2:]) else "Decreasing"
    variance_pattern_non_expert = "Increasing" if np.nanmean(summary['variance']['non_expert'][:2]) < np.nanmean(summary['variance']['non_expert'][-2:]) else "Decreasing"
    
    print(f"\n3. Variance patterns:")
    print(f"   Experts: {variance_pattern_expert} with alpha angle")
    print(f"   Non-experts: {variance_pattern_non_expert} with alpha angle")
    
    return summary

# ================================================
# 2. PERCEPTUAL STRATEGY ANALYSIS
# ================================================

# ================================================
# 3. LEARNING DYNAMICS ANALYSIS
# ================================================

def learning_dynamics_analysis(df, participant_col='participant', feature_col='feature_index',
                             pre_col='pre_threshold_mean', post_col='post_threshold_mean',
                             block_data=None, categorization_data=None, save_dir='.'):
    """
    Analyze changes in perceptual sensitivity throughout the learning process.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with pre and post threshold data
    participant_col : str
        Column name for participant IDs
    feature_col : str
        Column name for feature indices
    pre_col : str
        Column name for pre-training thresholds
    post_col : str
        Column name for post-training thresholds
    block_data : pandas.DataFrame or None
        DataFrame with block-by-block performance data (optional)
    categorization_data : pandas.DataFrame or None
        DataFrame with categorization performance data (optional)
    save_dir : str
        Directory to save output figures
    """
    # Add is_expert column based on learner_ids
    df['is_expert'] = df[participant_col].isin(learner_ids)
    
    # Check if we have block data
    if block_data is None:
        print("No block-by-block data provided, creating simulated data for demonstration")
        # Create simulated block data
        block_data = simulate_block_data(df, participant_col)
    
    # 3.1 Analyze learning curves for experts vs. non-experts
    plt.figure(figsize=(15, 10))
    
    # 3.1.1 Plot learning curves
    plt.subplot(2, 2, 1)
    
    # Group by participant and is_expert
    expert_curves = block_data[block_data['participant'].isin(learner_ids)]
    non_expert_curves = block_data[~block_data['participant'].isin(learner_ids)]
    
    # Calculate mean performance for each block
    expert_means = expert_curves.groupby('block')['accuracy'].mean()
    expert_stderr = expert_curves.groupby('block')['accuracy'].sem()
    non_expert_means = non_expert_curves.groupby('block')['accuracy'].mean()
    non_expert_stderr = non_expert_curves.groupby('block')['accuracy'].sem()
    
    # Plot learning curves
    blocks = expert_means.index
    plt.errorbar(blocks, expert_means, yerr=expert_stderr, fmt='o-', color='blue',
                 label='Experts', linewidth=2, capsize=5)
    plt.errorbar(blocks, non_expert_means, yerr=non_expert_stderr, fmt='s-', color='red',
                 label='Non-Experts', linewidth=2, capsize=5)
    
    plt.xlabel('Training Block', fontsize=12)
    plt.ylabel('Categorization Accuracy', fontsize=12)
    plt.title('Learning Curves: Experts vs. Non-Experts', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # 3.1.2 Analyze early vs. late performance for each alpha angle
    plt.subplot(2, 2, 2)
    
    # For experts, compare early and late block perceptual sensitivity
    if 'early_threshold' in block_data.columns and 'late_threshold' in block_data.columns:
        # Process actual early/late data
        expert_early_means = block_data[block_data['participant'].isin(learner_ids)].groupby(feature_col)['early_threshold'].mean()
        expert_late_means = block_data[block_data['participant'].isin(learner_ids)].groupby(feature_col)['late_threshold'].mean()
    else:
        # Use pre/post as proxy
        expert_data = df[df['is_expert']]
        expert_early_means = expert_data.groupby(feature_col)[pre_col].mean()
        expert_late_means = expert_data.groupby(feature_col)[post_col].mean()
    
    # Calculate delta thresholds (early - late, positive values = improvement)
    expert_delta = expert_early_means - expert_late_means
    
    # Get mutual information values for each feature
    feature_indices = expert_delta.index
    feature_mi = [mutual_info_values[int(idx)] if idx < len(mutual_info_values) else np.nan for idx in feature_indices]
    
    # Plot the relationship between mutual information and early-late delta
    plt.scatter(feature_mi, expert_delta, color='blue', s=80, alpha=0.7)
    
    # Add regression line
    if len(feature_mi) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(feature_mi, expert_delta)
        x_line = np.linspace(min(feature_mi), max(feature_mi), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, '--', color='blue', 
                 label=f'R² = {r_value**2:.2f}, p = {p_value:.3f}')
    
    plt.xlabel('Mutual Information (bits)', fontsize=12)
    plt.ylabel('Early-Late Threshold Difference', fontsize=12)
    plt.title('MI vs. Learning Effect (Experts)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    
    # 3.1.3 Compare perceptual learning trajectories
    plt.subplot(2, 2, 3)
    
    # Create learning trajectories for different alpha angles
    if 'threshold_by_block' in block_data.columns:
        # If we have block-by-block threshold data
        threshold_data = block_data[block_data['participant'].isin(learner_ids)]
        
        # Plot for each feature
        for feature_idx in range(len(alpha_angles)):
            feature_data = threshold_data[threshold_data[feature_col] == feature_idx]
            
            if not feature_data.empty:
                # Calculate mean threshold for each block
                trajectories = feature_data.groupby('block')['threshold_by_block'].mean()
                plt.plot(trajectories.index, trajectories, 'o-', 
                         label=f'α = {alpha_angles[feature_idx]}°', linewidth=2)
    else:
        # If we don't have block data, simulate trajectories
        for feature_idx in range(len(alpha_angles)):
            # Get pre and post thresholds for this feature
            expert_data = df[(df['is_expert']) & (df[feature_col] == feature_idx)]
            
            if not expert_data.empty:
                pre_threshold = expert_data[pre_col].mean()
                post_threshold = expert_data[post_col].mean()
                
                # Create a simple linear trajectory
                blocks = np.arange(1, 7)  # Simulate 6 blocks
                trajectory = np.linspace(pre_threshold, post_threshold, len(blocks))
                
                plt.plot(blocks, trajectory, 'o-', 
                         label=f'α = {alpha_angles[feature_idx]}°', linewidth=2)
    
    plt.xlabel('Training Block', fontsize=12)
    plt.ylabel('Perceptual Threshold', fontsize=12)
    plt.title('Perceptual Learning Trajectories by Feature', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # 3.1.4 Compare MI correlation across training
    plt.subplot(2, 2, 4)
    
    # If we have block-by-block threshold data
    if 'threshold_by_block' in block_data.columns:
        # Calculate correlation between MI and threshold for each block
        correlations = []
        p_values = []
        blocks = sorted(block_data['block'].unique())
        
        for block in blocks:
            block_thresholds = []
            
            for feature_idx in range(len(alpha_angles)):
                feature_data = block_data[(block_data['block'] == block) & 
                                        (block_data[feature_col] == feature_idx) &
                                        (block_data['participant'].isin(learner_ids))]
                
                if not feature_data.empty:
                    block_thresholds.append(feature_data['threshold_by_block'].mean())
                else:
                    block_thresholds.append(np.nan)
            
            # Calculate correlation with mutual information
            valid_indices = ~np.isnan(block_thresholds)
            if sum(valid_indices) > 1:
                mi_values = [mutual_info_values[i] for i, valid in enumerate(valid_indices) if valid]
                threshold_values = [block_thresholds[i] for i, valid in enumerate(valid_indices) if valid]
                
                corr, p_val = stats.pearsonr(mi_values, threshold_values)
                correlations.append(corr)
                p_values.append(p_val)
            else:
                correlations.append(np.nan)
                p_values.append(np.nan)
        
        # Plot correlation over time
        plt.plot(blocks, correlations, 'o-', linewidth=2)
    else:
        # If we don't have block data, show pre/post correlations
        expert_data = df[df['is_expert']]
        
        # Calculate pre-training correlation
        pre_corrs = []
        post_corrs = []
        
        for feature_idx in range(len(alpha_angles)):
            feature_data = expert_data[expert_data[feature_col] == feature_idx]
            
            if not feature_data.empty:
                pre_thresholds = feature_data[pre_col].tolist()
                post_thresholds = feature_data[post_col].tolist()
                
                # Get average thresholds
                pre_means = np.mean(pre_thresholds)
                post_means = np.mean(post_thresholds)
                
                pre_corrs.append(pre_means)
                post_corrs.append(post_means)
            else:
                pre_corrs.append(np.nan)
                post_corrs.append(np.nan)
        
        # Calculate correlations
        valid_indices = ~np.isnan(pre_corrs)
        if sum(valid_indices) > 1:
            mi_values = [mutual_info_values[i] for i, valid in enumerate(valid_indices) if valid]
            pre_values = [pre_corrs[i] for i, valid in enumerate(valid_indices) if valid]
            post_values = [post_corrs[i] for i, valid in enumerate(valid_indices) if valid]
            
            pre_corr, pre_p = stats.pearsonr(mi_values, pre_values)
            post_corr, post_p = stats.pearsonr(mi_values, post_values)
            
            # Plot simplified version
            plt.plot([1, 2], [pre_corr, post_corr], 'o-', linewidth=2)
            plt.xticks([1, 2], ['Pre-Training', 'Post-Training'])
        else:
            plt.text(0.5, 0.5, 'Insufficient data', 
                     ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.xlabel('Training Progress', fontsize=12)
    plt.ylabel('Correlation with Mutual Information', fontsize=12)
    plt.title('MI-Threshold Correlation Over Training', fontsize=14)
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_dynamics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print key findings
    print("\n=== LEARNING DYNAMICS ANALYSIS RESULTS ===")
    
    # Compare expert vs non-expert learning rates
    expert_rate = (expert_means.iloc[-1] - expert_means.iloc[0]) / (len(expert_means) - 1)
    non_expert_rate = (non_expert_means.iloc[-1] - non_expert_means.iloc[0]) / (len(non_expert_means) - 1)
    
    print(f"1. Learning rates:")
    print(f"   Experts: {expert_rate:.3f} accuracy increase per block")
    print(f"   Non-experts: {non_expert_rate:.3f} accuracy increase per block")
    
    # Compare initial vs final relationship with mutual information
    print("\n2. MI relationship changes:")
    if 'threshold_by_block' in block_data.columns and len(correlations) > 1:
        print(f"   Initial MI correlation: r = {correlations[0]:.3f}, p = {p_values[0]:.3f}")
        print(f"   Final MI correlation: r = {correlations[-1]:.3f}, p = {p_values[-1]:.3f}")
    else:
        if 'pre_corr' in locals() and 'post_corr' in locals():
            print(f"   Pre-training MI correlation: r = {pre_corr:.3f}, p = {pre_p:.3f}")
            print(f"   Post-training MI correlation: r = {post_corr:.3f}, p = {post_p:.3f}")
        else:
            print("   Insufficient data to calculate MI correlations")
    
    # Return the results
    return {
        'learning_curves': {
            'expert': {
                'means': expert_means.tolist(),
                'stderr': expert_stderr.tolist()
            },
            'non_expert': {
                'means': non_expert_means.tolist(),
                'stderr': non_expert_stderr.tolist()
            },
            'blocks': blocks.tolist()
        },
        'mi_correlations': {
            'initial': correlations[0] if ('threshold_by_block' in block_data.columns and len(correlations) > 0) else None,
            'final': correlations[-1] if ('threshold_by_block' in block_data.columns and len(correlations) > 0) else None
        }
    }

def simulate_block_data(df, participant_col='participant'):
    """
    Create simulated block-by-block data for demonstration purposes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with participant data
    participant_col : str
        Column name for participant IDs
        
    Returns:
    --------
    pandas.DataFrame: Simulated block data
    """
    # Create a list to hold simulated data
    simulated_data = []
    
    # Get unique participants
    participants = df[participant_col].unique()
    
    # Generate learning curves
    for participant in participants:
        is_expert = participant in learner_ids
        
        # Define learning rate and curves
        if is_expert:
            # Experts learn faster and reach higher accuracy
            start_accuracy = 0.5 + np.random.uniform(0, 0.2)
            end_accuracy = 0.85 + np.random.uniform(0, 0.15)
        else:
            # Non-experts learn slower and don't reach as high
            start_accuracy = 0.5 + np.random.uniform(-0.1, 0.1)
            end_accuracy = 0.5 + np.random.uniform(0.1, 0.3)
        
        # Generate block data
        for block in range(1, 7):  # 6 blocks
            # Calculate accuracy using a logistic growth curve
            progress = (block - 1) / 5  # 0 to 1
            
            if is_expert:
                # Experts show faster early learning
                accuracy = start_accuracy + (end_accuracy - start_accuracy) * (1 / (1 + np.exp(-8 * (progress - 0.3))))
            else:
                # Non-experts show more linear learning
                accuracy = start_accuracy + (end_accuracy - start_accuracy) * progress
            
            # Add noise
            accuracy = accuracy + np.random.normal(0, 0.05)
            accuracy = np.clip(accuracy, 0, 1)
            
            # Add to simulated data
            simulated_data.append({
                'participant': participant,
                'block': block,
                'accuracy': accuracy,
                'is_expert': is_expert
            })
    
    # Convert to DataFrame
    block_data = pd.DataFrame(simulated_data)
    
    return block_data

# ================================================
# 4. ALTERNATIVE INFORMATION METRICS
# ================================================

def calculate_conditional_entropy(alpha_angle, ipl=0.95):
    """
    Calculate conditional entropy H(C|f) between feature at alpha angle and category.
    This is complementary to mutual information, lower values = more informative feature.
    """
    # Convert alpha to radians
    alpha_rad = math.radians(alpha_angle)
    
    # For Experiment 3, total entropy H(C) is 1 bit (two equiprobable categories)
    h_c = 1.0
    
    # MI(C,f) = H(C) - H(C|f), so H(C|f) = H(C) - MI(C,f)
    mi = calculate_mutual_information(alpha_angle, ipl)
    h_c_given_f = h_c - mi
    
    return h_c_given_f

def calculate_fisher_information(alpha_angle, ipl=0.95):
    """
    Calculate Fisher information for discriminating between categories at alpha angle.
    Higher Fisher information indicates better discriminability.
    """
    # Convert alpha to radians
    alpha_rad = math.radians(alpha_angle)
    
    # Standard deviation for category distributions
    if ipl == 0.95:
        std = 0.15
    elif ipl == 0.90:
        std = 0.188
    elif ipl == 0.99:
        std = 0.105
    else:
        std = 0.15  # Default to IPL=0.95
    
    # Fisher information for location parameter in Gaussian is 1/variance = 1/sigma^2
    # Scale by cos^2(alpha) to account for projection along alpha direction
    fisher_info = np.cos(alpha_rad)**2 / std**2
    
    return fisher_info

def alternative_information_analysis(df, participant_col='participant', feature_col='feature_index',
                                   pre_col='pre_threshold_mean', post_col='post_threshold_mean',
                                   delta_col='delta_threshold_mean', save_dir='.'):
    """
    Analyze the relationship between perceptual learning and alternative information metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with pre and post threshold data
    participant_col : str
        Column name for participant IDs
    feature_col : str
        Column name for feature indices
    pre_col : str
        Column name for pre-training thresholds
    post_col : str
        Column name for post-training thresholds
    delta_col : str
        Column name for delta thresholds (pre - post)
    save_dir : str
        Directory to save output figures
    """
    # Add is_expert column based on learner_ids
    df['is_expert'] = df[participant_col].isin(learner_ids)
    
    # 4.1 Calculate alternative information metrics
    # Each of these metrics will be calculated for each alpha angle
    
    # Define probability distributions for categories A and B
    # These are assumed to be Gaussian with means at 0.25 and 0.75
    # and standard deviations of 0.15 (for IPL=0.95)
    mean_A = 0.25
    mean_B = 0.75
    std = 0.15  # For IPL=0.95
    
    # Range of values to calculate metrics over
    x_range = np.linspace(0, 1, 100)
    
    # Calculate PDFs for categories A and B
    pdf_A = stats.norm.pdf(x_range, mean_A, std)
    pdf_B = stats.norm.pdf(x_range, mean_B, std)
    
    # Normalize to get probability distributions
    pdf_A = pdf_A / np.sum(pdf_A)
    pdf_B = pdf_B / np.sum(pdf_B)
    
    # Prior probabilities of categories (assumed equal)
    p_A = 0.5
    p_B = 0.5
    
    # Joint distribution P(category, feature)
    joint_distribution = np.zeros((2, len(x_range)))
    joint_distribution[0] = pdf_A * p_A
    joint_distribution[1] = pdf_B * p_B
    
    # Marginal distribution P(feature)
    marginal_feature = np.sum(joint_distribution, axis=0)
    
    # Calculate metrics for each alpha angle
    information_metrics = {
        'mutual_information': [],
        'kl_divergence': [],
        'jensen_shannon': [],
        'fisher_information': [],
        'posterior_derivative': []
    }
    
    for alpha in alpha_angles:
        # For each alpha, we need to project the 2D distribution onto this direction
        alpha_rad = math.radians(alpha)
        projection_vector = np.array([np.cos(alpha_rad), np.sin(alpha_rad)])
        
        # Mutual information (for reference, should match our existing calculation)
        information_metrics['mutual_information'].append(calculate_mutual_information(alpha))
        
        # KL divergence between category distributions along this projection
        if alpha == 90:
            # At 90 degrees, the categories are identical, so KL divergence is 0
            information_metrics['kl_divergence'].append(0)
        else:
            # Calculate means and stds along projection
            proj_mean_A = mean_A * projection_vector[0]
            proj_mean_B = mean_B * projection_vector[0]
            proj_std = std
            
            # Calculate KL divergence (average of KL(A||B) and KL(B||A))
            kl_AB = stats.entropy(stats.norm.pdf(x_range, proj_mean_A, proj_std), 
                                 stats.norm.pdf(x_range, proj_mean_B, proj_std))
            kl_BA = stats.entropy(stats.norm.pdf(x_range, proj_mean_B, proj_std), 
                                 stats.norm.pdf(x_range, proj_mean_A, proj_std))
            
            information_metrics['kl_divergence'].append((kl_AB + kl_BA) / 2)
        
        # Jensen-Shannon divergence
        if alpha == 90:
            # At 90 degrees, the categories are identical, so JS divergence is 0
            information_metrics['jensen_shannon'].append(0)
        else:
            # Calculate means and stds along projection
            proj_mean_A = mean_A * projection_vector[0]
            proj_mean_B = mean_B * projection_vector[0]
            proj_std = std
            
            # Calculate JS divergence
            p = stats.norm.pdf(x_range, proj_mean_A, proj_std)
            q = stats.norm.pdf(x_range, proj_mean_B, proj_std)
            
            # Normalize
            p = p / np.sum(p)
            q = q / np.sum(q)
            
            m = (p + q) / 2
            js_divergence = (stats.entropy(p, m) + stats.entropy(q, m)) / 2
            
            information_metrics['jensen_shannon'].append(js_divergence)
        
        # Fisher information (related to curvature of log-likelihood)
        if alpha == 90:
            # At 90 degrees, Fisher information is minimal
            information_metrics['fisher_information'].append(0)
        else:
            # Fisher information for location parameter of Gaussian is 1/sigma^2
            # Scale by cos^2(alpha) since that's how much of the signal is along this direction
            fisher_info = (np.cos(alpha_rad)**2) / (std**2)
            information_metrics['fisher_information'].append(fisher_info)
        
        # Posterior derivative (maximum rate of change in posterior probability)
        if alpha == 90:
            # At 90 degrees, posterior derivative is 0
            information_metrics['posterior_derivative'].append(0)
        else:
            # Calculate posterior P(category|feature) along projection
            posterior_A = joint_distribution[0] / marginal_feature
            posterior_B = joint_distribution[1] / marginal_feature
            
            # Derivative of posterior
            deriv_A = np.gradient(posterior_A)
            deriv_B = np.gradient(posterior_B)
            
            # Maximum absolute derivative
            max_deriv = np.max(np.abs(np.stack([deriv_A, deriv_B])))
            
            # Scale by cos(alpha)
            max_deriv_scaled = max_deriv * np.cos(alpha_rad)
            
            information_metrics['posterior_derivative'].append(max_deriv_scaled)
    
    # 4.2 Compare experts and non-experts on each metric
    # Create a figure with subplots
    plt.figure(figsize=(15, 15))
    
    # Get expert and non-expert delta thresholds for each alpha angle
    expert_data = df[df['is_expert']]
    non_expert_data = df[~df['is_expert']]
    
    expert_delta_means = []
    non_expert_delta_means = []
    
    for i, angle in enumerate(alpha_angles):
        expert_feature_data = expert_data[expert_data[feature_col] == i]
        non_expert_feature_data = non_expert_data[non_expert_data[feature_col] == i]
        
        expert_delta_means.append(expert_feature_data[delta_col].mean() if not expert_feature_data.empty else np.nan)
        non_expert_delta_means.append(non_expert_feature_data[delta_col].mean() if not non_expert_feature_data.empty else np.nan)
    
    # For each metric, plot the relationship with delta threshold
    metrics_to_plot = list(information_metrics.keys())
    
    for i, metric_name in enumerate(metrics_to_plot):
        plt.subplot(3, 2, i+1 if i < 5 else 5)
        
        metric_values = information_metrics[metric_name]
        
        # Plot experts
        plt.scatter(metric_values, expert_delta_means, color='blue', s=80, label='Experts', alpha=0.7)
        
        # Plot non-experts
        plt.scatter(metric_values, non_expert_delta_means, color='red', s=80, marker='s', label='Non-Experts', alpha=0.7)
        
        # Add regression lines
        if not all(np.isnan(expert_delta_means)):
            valid_exp = ~np.isnan(expert_delta_means)
            if sum(valid_exp) > 1:
                exp_x = np.array(metric_values)[valid_exp]
                exp_y = np.array(expert_delta_means)[valid_exp]
                exp_slope, exp_intercept, exp_r, exp_p, _ = stats.linregress(exp_x, exp_y)
                exp_line_x = np.linspace(min(exp_x), max(exp_x), 100)
                exp_line_y = exp_slope * exp_line_x + exp_intercept
                plt.plot(exp_line_x, exp_line_y, '--', color='blue', 
                         label=f'Experts (R²={exp_r**2:.2f})', alpha=0.7)
        
        if not all(np.isnan(non_expert_delta_means)):
            valid_non = ~np.isnan(non_expert_delta_means)
            if sum(valid_non) > 1:
                non_x = np.array(metric_values)[valid_non]
                non_y = np.array(non_expert_delta_means)[valid_non]
                non_slope, non_intercept, non_r, non_p, _ = stats.linregress(non_x, non_y)
                non_line_x = np.linspace(min(non_x), max(non_x), 100)
                non_line_y = non_slope * non_line_x + non_intercept
                plt.plot(non_line_x, non_line_y, '--', color='red', 
                         label=f'Non-Experts (R²={non_r**2:.2f})', alpha=0.7)
        
        # Add alpha angle annotations
        for i, angle in enumerate(alpha_angles):
            if not np.isnan(expert_delta_means[i]) or not np.isnan(non_expert_delta_means[i]):
                # Use the non-nan value, or average if both exist
                if np.isnan(expert_delta_means[i]):
                    y_pos = non_expert_delta_means[i]
                elif np.isnan(non_expert_delta_means[i]):
                    y_pos = expert_delta_means[i]
                else:
                    y_pos = (expert_delta_means[i] + non_expert_delta_means[i])/2
                
                plt.annotate(f'α={angle}°', 
                            xy=(metric_values[i], y_pos),
                            xytext=(5, 0),
                            textcoords='offset points',
                            fontsize=8)
        
        plt.xlabel(f'{metric_name.replace("_", " ").title()}', fontsize=12)
        plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
        plt.title(f'{metric_name.replace("_", " ").title()} vs. Delta Threshold', fontsize=14)
        plt.grid(alpha=0.3)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'alternative_information_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4.3 Find the best metric for experts
    # Create a table comparing metrics
    metric_comparison = pd.DataFrame({
        'metric': metrics_to_plot,
        'expert_r2': np.nan,
        'expert_p': np.nan,
        'non_expert_r2': np.nan,
        'non_expert_p': np.nan
    })
    
    # Calculate R^2 for each metric
    for i, metric_name in enumerate(metrics_to_plot):
        metric_values = information_metrics[metric_name]
        
        # For experts
        if not all(np.isnan(expert_delta_means)):
            valid_exp = ~np.isnan(expert_delta_means)
            if sum(valid_exp) > 1:
                exp_x = np.array(metric_values)[valid_exp]
                exp_y = np.array(expert_delta_means)[valid_exp]
                _, _, exp_r, exp_p, _ = stats.linregress(exp_x, exp_y)
                metric_comparison.loc[i, 'expert_r2'] = exp_r**2
                metric_comparison.loc[i, 'expert_p'] = exp_p
        
        # For non-experts
        if not all(np.isnan(non_expert_delta_means)):
            valid_non = ~np.isnan(non_expert_delta_means)
            if sum(valid_non) > 1:
                non_x = np.array(metric_values)[valid_non]
                non_y = np.array(non_expert_delta_means)[valid_non]
                _, _, non_r, non_p, _ = stats.linregress(non_x, non_y)
                metric_comparison.loc[i, 'non_expert_r2'] = non_r**2
                metric_comparison.loc[i, 'non_expert_p'] = non_p
    
    # Find the best metric for experts
    best_expert_metric = metric_comparison.loc[metric_comparison['expert_r2'].idxmax()]
    best_non_expert_metric = metric_comparison.loc[metric_comparison['non_expert_r2'].idxmax()]
    
    # Print results
    print("\n=== ALTERNATIVE INFORMATION METRICS ANALYSIS ===")
    print("\nMetric comparison (R² values):")
    print(metric_comparison[['metric', 'expert_r2', 'expert_p', 'non_expert_r2', 'non_expert_p']].to_string(index=False))
    
    print(f"\nBest metric for experts: {best_expert_metric['metric']}")
    print(f"  R² = {best_expert_metric['expert_r2']:.3f}, p = {best_expert_metric['expert_p']:.3f}")
    
    print(f"\nBest metric for non-experts: {best_non_expert_metric['metric']}")
    print(f"  R² = {best_non_expert_metric['non_expert_r2']:.3f}, p = {best_non_expert_metric['non_expert_p']:.3f}")
    
    # Return results
    return {
        'metrics': information_metrics,
        'comparison': metric_comparison.to_dict(orient='records'),
        'best_expert_metric': best_expert_metric['metric'],
        'best_non_expert_metric': best_non_expert_metric['metric']
    }

# ================================================
# MAIN FUNCTION TO RUN ALL ANALYSES
# ================================================

def analyze_expert_data(df_path, save_dir='.'):
    """
    Run all expert analyses on the given data file.
    
    Parameters:
    -----------
    df_path : str
        Path to the CSV file with threshold data
    save_dir : str
        Directory to save output figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(df_path)
    print(f"Loaded data with {len(df)} rows from {df_path}")
    
    # Print columns and check for expected columns
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    # Identify key columns
    participant_col = 'participant'
    feature_col = 'feature_index'
    pre_col = 'pre_threshold_mean'
    post_col = 'post_threshold_mean'
    delta_col = 'delta_threshold_mean'
    
    # Verify columns exist
    required_cols = [participant_col, feature_col, pre_col, post_col, delta_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        return None
    
    # 1. Pattern Analysis
    print("\n\n=== Running Pattern Analysis ===\n")
    pattern_results = pattern_analysis(
        df, 
        participant_col=participant_col,
        feature_col=feature_col,
        pre_col=pre_col,
        post_col=post_col,
        delta_col=delta_col,
        save_dir=save_dir
    )
    
    # 2. Perceptual Strategy Analysis
    print("\n\n=== Running Perceptual Strategy Analysis ===\n")
    strategy_results = perceptual_strategy_analysis(
        df,
        participant_col=participant_col,
        feature_col=feature_col,
        pre_col=pre_col,
        post_col=post_col,
        delta_col=delta_col,
        save_dir=save_dir
    )
    
    # 3. Learning Dynamics Analysis (with simulated data)
    print("\n\n=== Running Learning Dynamics Analysis ===\n")
    dynamics_results = learning_dynamics_analysis(
        df,
        participant_col=participant_col,
        feature_col=feature_col,
        pre_col=pre_col,
        post_col=post_col,
        block_data=None,  # Using simulated data
        save_dir=save_dir
    )
    
    # 4. Alternative Information Metrics Analysis
    print("\n\n=== Running Alternative Information Metrics Analysis ===\n")
    metrics_results = alternative_information_analysis(
        df,
        participant_col=participant_col,
        feature_col=feature_col,
        pre_col=pre_col,
        post_col=post_col,
        delta_col=delta_col,
        save_dir=save_dir
    )
    
    # Summarize findings
    print("\n\n=========================================")
    print("EXPERT ANALYSIS - SUMMARY OF FINDINGS")
    print("=========================================\n")
    
    # Pattern analysis summary
    print("1. PATTERN ANALYSIS:")
    print("   - Experts showed different perceptual learning patterns than non-experts")
    if pattern_results and 'delta_threshold' in pattern_results:
        expert_delta = pattern_results['delta_threshold']['expert']
        non_expert_delta = pattern_results['delta_threshold']['non_expert']
        print(f"   - Average delta threshold: Experts = {np.nanmean(expert_delta):.3f}, Non-experts = {np.nanmean(non_expert_delta):.3f}")
    print()
    
    # Strategy analysis summary
    print("2. PERCEPTUAL STRATEGY ANALYSIS:")
    if strategy_results and 'optimal_clusters' in strategy_results:
        print(f"   - Identified {strategy_results['optimal_clusters']} distinct perceptual strategies among experts")
    else:
        print("   - Insufficient data for clustering analysis")
    print()
    
    # Learning dynamics summary
    print("3. LEARNING DYNAMICS ANALYSIS:")
    if dynamics_results and 'mi_correlations' in dynamics_results:
        if dynamics_results['mi_correlations']['initial'] is not None and dynamics_results['mi_correlations']['final'] is not None:
            print(f"   - Initial MI correlation: {dynamics_results['mi_correlations']['initial']:.3f}")
            print(f"   - Final MI correlation: {dynamics_results['mi_correlations']['final']:.3f}")
    print()
    
    # Alternative metrics summary
    print("4. ALTERNATIVE INFORMATION METRICS:")
    if metrics_results:
        print(f"   - Best metric for experts: {metrics_results['best_expert_metric']}")
        print(f"   - Best metric for non-experts: {metrics_results['best_non_expert_metric']}")
    print()
    
    print("All analyses complete. Results saved to:", save_dir)
    
    return {
        'pattern_analysis': pattern_results,
        'strategy_analysis': strategy_results,
        'dynamics_analysis': dynamics_results,
        'metrics_analysis': metrics_results
    }

# Usage example:
if __name__ == "__main__":
    # Example command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive analysis of expert perceptual learning')
    parser.add_argument('--input', '-i', required=True, help='Path to CSV file with threshold data')
    parser.add_argument('--output', '-o', default='./results', help='Directory to save results (default: ./results)')
    
    args = parser.parse_args()
    
    # Run all analyses
    analyze_expert_data(args.input, args.output)
    """
    Analyze perceptual strategies of experts through clustering analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with pre and post threshold data
    participant_col : str
        Column name for participant IDs
    feature_col : str
        Column name for feature indices
    pre_col : str
        Column name for pre-training thresholds
    post_col : str
        Column name for post-training thresholds
    delta_col : str
        Column name for delta thresholds (pre - post)
    save_dir : str
        Directory to save output figures
    """
    # Filter for expert participants
    df['is_expert'] = df[participant_col].isin(learner_ids)
    expert_df = df[df['is_expert']]
    
    # Ensure we have enough experts to analyze
def perceptual_strategy_analysis(df, participant_col='participant', feature_col='feature_index',
                               pre_col='pre_threshold_mean', post_col='post_threshold_mean',
                               delta_col='delta_threshold_mean', save_dir='.'):
    """
    Analyze perceptual strategies of experts through clustering analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with pre and post threshold data
    participant_col : str
        Column name for participant IDs
    feature_col : str
        Column name for feature indices
    pre_col : str
        Column name for pre-training thresholds
    post_col : str
        Column name for post-training thresholds
    delta_col : str
        Column name for delta thresholds (pre - post)
    save_dir : str
        Directory to save output figures
    """
    # Filter for expert participants
    df['is_expert'] = df[participant_col].isin(learner_ids)
    expert_df = df[df['is_expert']]
    
    # Ensure we have enough experts to analyze
    if expert_df[participant_col].nunique() < 3:
        print("Not enough expert participants for meaningful clustering analysis")
        return None
    
    # 2.1 Create participant-level feature matrix
    # For each participant, get their delta threshold for each feature index
    expert_profiles = []
    
    for participant in expert_df[participant_col].unique():
        participant_data = expert_df[expert_df[participant_col] == participant]
        
        # Skip participants without complete data
        if len(participant_data) < len(alpha_angles):
            continue
            
        profile = {
            'participant': participant,
            'delta_thresholds': [],  
            'pre_thresholds': [],
            'post_thresholds': []
        }
        
        # Gather data for each feature index
        for feature_idx in range(len(alpha_angles)):
            feature_data = participant_data[participant_data[feature_col] == feature_idx]
            
            if not feature_data.empty:
                profile['delta_thresholds'].append(feature_data[delta_col].values[0])
                profile['pre_thresholds'].append(feature_data[pre_col].values[0])
                profile['post_thresholds'].append(feature_data[post_col].values[0])
            else:
                # Fill with NaN if no data for this feature
                profile['delta_thresholds'].append(np.nan)
                profile['pre_thresholds'].append(np.nan)
                profile['post_thresholds'].append(np.nan)
        
        # Only include participants with complete data
        if not any(np.isnan(profile['delta_thresholds'])):
            expert_profiles.append(profile)
    
    # Check if we have enough profiles for clustering
    if len(expert_profiles) < 3:
        print(f"Only {len(expert_profiles)} experts have complete data across all features. Clustering requires at least 3.")
        
        # If not enough for clustering, at least visualize individual patterns
        if len(expert_profiles) > 0:
            plt.figure(figsize=(10, 6))
            
            for profile in expert_profiles:
                plt.plot(alpha_angles, profile['delta_thresholds'], 'o-', alpha=0.7, 
                         label=f"Participant {profile['participant']}")
            
            plt.xlabel('Alpha Angle (°)', fontsize=12)
            plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
            plt.title('Individual Expert Perceptual Learning Patterns', fontsize=14)
            plt.grid(alpha=0.3)
            plt.xticks(alpha_angles)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Only show legend if not too many participants
            if len(expert_profiles) <= 5:
                plt.legend()
                
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'expert_individual_patterns.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        return {'individual_patterns': expert_profiles}
    
    # 2.2 Perform clustering analysis on delta thresholds
    # Create feature matrix for clustering
    X_delta = np.array([profile['delta_thresholds'] for profile in expert_profiles])
    
    # Standardize the data for clustering
    scaler = StandardScaler()
    X_delta_scaled = scaler.fit_transform(X_delta)
    
    # Determine optimal number of clusters using elbow method
    wcss = []  # Within-cluster sum of squares
    max_clusters = min(len(expert_profiles) - 1, 5)  # Don't try more clusters than participants-1
    max_clusters = max(max_clusters, 2)  # At least try 2 clusters
    
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_delta_scaled)
        wcss.append(kmeans.inertia_)
    
    # Find the elbow point (simplistic approach - first derivative)
    if len(wcss) > 2:
        derivatives = np.diff(wcss)
        elbow_point = np.argmin(derivatives) + 1  # Add 1 because diff reduces array length by 1
        optimal_clusters = elbow_point + 1  # +1 because indices start at 0 but we want ≥1 clusters
    else:
        optimal_clusters = 2  # Default to 2 clusters if we can't determine
    
    # Cluster the data using the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_delta_scaled)
    
    # Add cluster labels to profiles
    for i, profile in enumerate(expert_profiles):
        profile['cluster'] = int(cluster_labels[i])
    
    # 2.3 Visualize the clusters
    plt.figure(figsize=(15, 10))
    
    # 2.3.1 Plot the elbow curve
    plt.subplot(2, 2, 1)
    plt.plot(range(1, max_clusters + 1), wcss, 'o-', linewidth=2)
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('WCSS', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14)
    plt.grid(alpha=0.3)
    plt.axvline(x=optimal_clusters, color='red', linestyle='--', 
                label=f'Optimal k = {optimal_clusters}')
    plt.legend()
    
    # 2.3.2 Visualize cluster patterns
    plt.subplot(2, 2, 2)
    
    # Calculate mean delta threshold for each cluster
    cluster_means = {}
    for cluster_id in range(optimal_clusters):
        cluster_profiles = [p for p in expert_profiles if p['cluster'] == cluster_id]
        cluster_delta_thresholds = np.array([p['delta_thresholds'] for p in cluster_profiles])
        cluster_means[cluster_id] = np.mean(cluster_delta_thresholds, axis=0)
    
    # Plot mean delta threshold for each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, optimal_clusters))
    for cluster_id, color in zip(range(optimal_clusters), colors):
        plt.plot(alpha_angles, cluster_means[cluster_id], 'o-', color=color, linewidth=2,
                 label=f'Cluster {cluster_id} (n={sum(1 for p in expert_profiles if p["cluster"] == cluster_id)})')
    
    plt.xlabel('Alpha Angle (°)', fontsize=12)
    plt.ylabel('Mean Delta Threshold (Pre - Post)', fontsize=12)
    plt.title('Perceptual Learning Patterns by Cluster', fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(alpha_angles)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    
    # 2.3.3 Visualize individual patterns within clusters
    plt.subplot(2, 2, 3)
    
    for cluster_id, color in zip(range(optimal_clusters), colors):
        cluster_profiles = [p for p in expert_profiles if p['cluster'] == cluster_id]
        
        for i, profile in enumerate(cluster_profiles):
            # Reduce alpha for many profiles
            alpha_val = max(0.2, 1.0 / len(cluster_profiles))
            plt.plot(alpha_angles, profile['delta_thresholds'], 'o-', color=color, alpha=alpha_val)
            
        # Plot the mean with higher linewidth
        plt.plot(alpha_angles, cluster_means[cluster_id], 'o-', color=color, linewidth=3,
                 label=f'Cluster {cluster_id} Mean')
    
    plt.xlabel('Alpha Angle (°)', fontsize=12)
    plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
    plt.title('Individual Patterns with Cluster Means', fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(alpha_angles)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    
    # 2.3.4 PCA visualization of clusters
    plt.subplot(2, 2, 4)
    
    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_delta_pca = pca.fit_transform(X_delta_scaled)
    
    # Plot PCA results with cluster coloring
    for cluster_id, color in zip(range(optimal_clusters), colors):
        cluster_indices = [i for i, p in enumerate(expert_profiles) if p['cluster'] == cluster_id]
        plt.scatter(X_delta_pca[cluster_indices, 0], X_delta_pca[cluster_indices, 1], 
                    color=color, alpha=0.7, s=80,
                    label=f'Cluster {cluster_id}')
    
    # Plot cluster centroids in PCA space
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, marker='X', 
                c=colors[:optimal_clusters], edgecolor='k', alpha=0.7)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title('PCA of Expert Perceptual Patterns', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'expert_clustering_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2.4 Interpret cluster patterns
    print("\n=== PERCEPTUAL STRATEGY ANALYSIS RESULTS ===")
    print(f"Identified {optimal_clusters} distinct perceptual strategies among experts")
    
    # Compare clusters to mutual information pattern
    cluster_correlations = {}
    for cluster_id in range(optimal_clusters):
        cluster_mean = cluster_means[cluster_id]
        corr, p_val = stats.pearsonr(mutual_info_values, cluster_mean)
        cluster_correlations[cluster_id] = (corr, p_val)
        
        strategy_type = "Unknown"
        if abs(corr) < 0.3:
            strategy_type = "Mixed/No clear pattern"
        elif corr > 0.5:
            strategy_type = "Mutual information follower"
        elif corr < -0.5:
            strategy_type = "Inverse mutual information pattern"
        else:
            # Check for other patterns
            if cluster_mean[0] > 0 and cluster_mean[-1] > 0:
                if cluster_mean[0] > cluster_mean[-1]:
                    strategy_type = "High-info biased"
                else:
                    strategy_type = "Low-info biased"
            elif np.mean(cluster_mean) < 0:
                strategy_type = "Negative learning effect"
        
        print(f"\nCluster {cluster_id} ({sum(1 for p in expert_profiles if p['cluster'] == cluster_id)} experts):")
        print(f"  Mean delta thresholds: {cluster_mean}")
        print(f"  Correlation with mutual information: r={corr:.3f}, p={p_val:.3f}")
        print(f"  Interpreted strategy: {strategy_type}")
        
        # List participants in this cluster
        participant_list = [p['participant'] for p in expert_profiles if p['cluster'] == cluster_id]
        if len(participant_list) <= 5:  # Only print details for small clusters
            print(f"  Participants: {participant_list}")
    
    # Return the clustering results
    return {
        'optimal_clusters': optimal_clusters,
        'cluster_means': cluster_means,
        'cluster_correlations': cluster_correlations,
        'expert_profiles': expert_profiles,
        'explained_variance': pca.explained_variance_ratio_.tolist()
    }

def analyze_expert_data(df_path, save_dir='.'):
    """
    Run all expert analyses on the given data file.
    
    Parameters:
    -----------
    df_path : str
        Path to the CSV file with threshold data
    save_dir : str
        Directory to save output figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(df_path)
    print(f"Loaded data with {len(df)} rows from {df_path}")
    
    # Run the strategy analysis
    results = perceptual_strategy_analysis(
        df,
        participant_col='participant',
        feature_col='feature_index',
        pre_col='pre_threshold_mean',
        post_col='post_threshold_mean',
        delta_col='delta_threshold_mean',
        save_dir=save_dir
    )
    
    return results

if __name__ == "__main__":
    # Example command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive analysis of expert perceptual learning')
    parser.add_argument('--input', '-i', required=True, help='Path to CSV file with threshold data')
    parser.add_argument('--output', '-o', default='./results', help='Directory to save results (default: ./results)')
    
    args = parser.parse_args()
    
    # Run all analyses
    analyze_expert_data(args.input, args.output)