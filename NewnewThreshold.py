import pandas as pd
import numpy as np
import os
import math
from psychopy.data import QuestHandler

def process_directory_with_features(directory_path):
    """
    Process directory to calculate thresholds for each feature index (0-4).
    """
    print(f"\nProcessing directory: {directory_path}")
    
    # Get all CSV files in the directory
    file_list = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    
    if not file_list:
        print(f"No CSV files found in {directory_path}")
        return None
    
    print(f"Found {len(file_list)} CSV files")
    
    # Parameters for QuestHandler
    startIntensity = 0.1
    startSD = 0.3
    pThreshold = 0.82
    beta = 3.5
    delta = 0.01 
    gamma = 0.5
    
    # Store results by feature index
    feature_results = {}
    for feature_idx in range(5):  # Feature indices 0-4
        feature_results[feature_idx] = []
    
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        
        try:
            df = pd.read_csv(file_path)
            
            # Check if the required columns exist
            required_columns = ["PreTest.intensity", "PreTest.response", 
                               "PostTest.intensity", "PostTest.response", "feature_index"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Skipping {file_name}: Missing columns {missing_columns}")
                continue
            
            # Process each feature index separately
            participant_id = file_name.replace(".csv", "")
            
            for feature_idx in range(5):  # Feature indices 0-4
                # Filter data for this feature index
                feature_data = df[df["feature_index"] == feature_idx]
                
                # Pre & Post-Test Data for this feature
                pre_intensities = feature_data["PreTest.intensity"].dropna().to_list()
                pre_responses = feature_data["PreTest.response"].dropna().astype(int).to_list()
                post_intensities = feature_data["PostTest.intensity"].dropna().to_list()
                post_responses = feature_data["PostTest.response"].dropna().astype(int).to_list()
                
                # Skip if insufficient data for this feature
                if len(pre_intensities) < 5 or len(post_intensities) < 5:
                    print(f"Insufficient data for participant {participant_id}, feature {feature_idx}")
                    continue
                
                # Last 5 trials data
                pre_intensities_last5 = pre_intensities[-5:]
                pre_responses_last5 = pre_responses[-5:]
                post_intensities_last5 = post_intensities[-5:]
                post_responses_last5 = post_responses[-5:]
                
                # Create QuestHandler for all Pre-Test data for this feature
                quest_pre = QuestHandler(
                    startVal=startIntensity,
                    startValSd=startSD,
                    pThreshold=pThreshold,
                    nTrials=len(pre_intensities),
                    beta=beta,
                    delta=delta,
                    gamma=gamma
                )
                quest_pre.importData(pre_intensities, pre_responses)
                
                # Create QuestHandler for all Post-Test data for this feature
                quest_post = QuestHandler(
                    startVal=startIntensity,
                    startValSd=startSD,
                    pThreshold=pThreshold,
                    nTrials=len(post_intensities),
                    beta=beta,
                    delta=delta,
                    gamma=gamma
                )
                quest_post.importData(post_intensities, post_responses)
                
                # Create QuestHandler for last 5 Pre-Test trials for this feature
                quest_pre_last5 = QuestHandler(
                    startVal=startIntensity,
                    startValSd=startSD,
                    pThreshold=pThreshold,
                    nTrials=len(pre_intensities_last5),
                    beta=beta,
                    delta=delta,
                    gamma=gamma
                )
                quest_pre_last5.importData(pre_intensities_last5, pre_responses_last5)
                
                # Create QuestHandler for last 5 Post-Test trials for this feature
                quest_post_last5 = QuestHandler(
                    startVal=startIntensity,
                    startValSd=startSD,
                    pThreshold=pThreshold,
                    nTrials=len(post_intensities_last5),
                    beta=beta,
                    delta=delta,
                    gamma=gamma
                )
                quest_post_last5.importData(post_intensities_last5, post_responses_last5)
                
                # Calculate thresholds for all trials
                pre_threshold_mean = quest_pre.mean()
                pre_threshold_mode = quest_pre.mode()
                pre_threshold_median = quest_pre.quantile(0.5)
                
                post_threshold_mean = quest_post.mean()
                post_threshold_mode = quest_post.mode()
                post_threshold_median = quest_post.quantile(0.5)
                
                # Calculate thresholds for last 5 trials
                pre_threshold_mean_last5 = quest_pre_last5.mean()
                pre_threshold_mode_last5 = quest_pre_last5.mode()
                pre_threshold_median_last5 = quest_pre_last5.quantile(0.5)
                
                post_threshold_mean_last5 = quest_post_last5.mean()
                post_threshold_mode_last5 = quest_post_last5.mode()
                post_threshold_median_last5 = quest_post_last5.quantile(0.5)
                
                # Calculate delta thresholds (Pre - Post) for all trials
                delta_threshold_mean = pre_threshold_mean - post_threshold_mean
                delta_threshold_mode = pre_threshold_mode - post_threshold_mode
                delta_threshold_median = pre_threshold_median - post_threshold_median
                
                # Calculate delta thresholds (Pre - Post) for last 5 trials
                delta_threshold_mean_last5 = pre_threshold_mean_last5 - post_threshold_mean_last5
                delta_threshold_mode_last5 = pre_threshold_mode_last5 - post_threshold_mode_last5
                delta_threshold_median_last5 = pre_threshold_median_last5 - post_threshold_median_last5
                
                # Store results for this feature
                feature_results[feature_idx].append({
                    "participant": participant_id,
                    "feature_index": feature_idx,
                    # All trials thresholds
                    "pre_threshold_mean": pre_threshold_mean,
                    "pre_threshold_mode": pre_threshold_mode,
                    "pre_threshold_median": pre_threshold_median,
                    "post_threshold_mean": post_threshold_mean,
                    "post_threshold_mode": post_threshold_mode,
                    "post_threshold_median": post_threshold_median,
                    "delta_threshold_mean": delta_threshold_mean,
                    "delta_threshold_mode": delta_threshold_mode,
                    "delta_threshold_median": delta_threshold_median,
                    # Last 5 trials thresholds
                    "pre_threshold_mean_last5": pre_threshold_mean_last5,
                    "pre_threshold_mode_last5": pre_threshold_mode_last5,
                    "pre_threshold_median_last5": pre_threshold_median_last5,
                    "post_threshold_mean_last5": post_threshold_mean_last5,
                    "post_threshold_mode_last5": post_threshold_mode_last5,
                    "post_threshold_median_last5": post_threshold_median_last5,
                    "delta_threshold_mean_last5": delta_threshold_mean_last5,
                    "delta_threshold_mode_last5": delta_threshold_mode_last5,
                    "delta_threshold_median_last5": delta_threshold_median_last5
                })
            
            print(f"Processed {file_name}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    # Save results for each feature index
    condition_name = os.path.basename(directory_path)
    
    # Create and save combined results dataframe
    all_results = []
    for feature_idx, results in feature_results.items():
        if results:  # Only add if we have results for this feature
            all_results.extend(results)
    
    if not all_results:
        print(f"No valid results found in {directory_path}")
        return None
    
    # Create DataFrame with all results
    results_df = pd.DataFrame(all_results)
    
    # Save the combined results to CSV
    output_filename = f"quest_thresholds_{condition_name}_by_feature.csv"
    output_path = os.path.join(directory_path, output_filename)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Also save individual feature CSV files for easier analysis
    for feature_idx, results in feature_results.items():
        if results:  # Only save if we have results for this feature
            feature_df = pd.DataFrame(results)
            feature_output_filename = f"quest_thresholds_{condition_name}_feature_{feature_idx}.csv"
            feature_output_path = os.path.join(directory_path, feature_output_filename)
            feature_df.to_csv(feature_output_path, index=False)
            print(f"Feature {feature_idx} results saved to {feature_output_path}")
    
    # Calculate and print summary statistics by feature
    print("\nSummary of Delta Thresholds by Feature Index:")
    for feature_idx, results in feature_results.items():
        if results:  # Only calculate if we have results for this feature
            feature_df = pd.DataFrame(results)
            avg_delta_mean = feature_df["delta_threshold_mean"].mean()
            avg_delta_mode = feature_df["delta_threshold_mode"].mean()
            avg_delta_median = feature_df["delta_threshold_median"].mean()
            
            # Last 5 trials
            avg_delta_mean_last5 = feature_df["delta_threshold_mean_last5"].mean()
            avg_delta_mode_last5 = feature_df["delta_threshold_mode_last5"].mean()
            avg_delta_median_last5 = feature_df["delta_threshold_median_last5"].mean()
            
            print(f"\nFeature Index {feature_idx}:")
            print(f"  Participants analyzed: {len(feature_df)}")
            print(f"  Average Delta Thresholds (All Trials):")
            print(f"    Mean method: {avg_delta_mean:.4f}")
            print(f"    Mode method: {avg_delta_mode:.4f}")
            print(f"    Median method: {avg_delta_median:.4f}")
            
            print(f"  Average Delta Thresholds (Last 5 Trials):")
            print(f"    Mean method: {avg_delta_mean_last5:.4f}")
            print(f"    Mode method: {avg_delta_mode_last5:.4f}")
            print(f"    Median method: {avg_delta_median_last5:.4f}")
    
    return results_df

def plot_feature_results(directory_path):
    """
    Plot delta thresholds by feature index
    """
    import matplotlib.pyplot as plt
    
    condition_name = os.path.basename(directory_path)
    combined_file = os.path.join(directory_path, f"quest_thresholds_{condition_name}_by_feature.csv")
    
    if not os.path.exists(combined_file):
        print(f"Results file not found: {combined_file}")
        return
    
    # Load the combined results
    df = pd.read_csv(combined_file)
    
    # Create a figure for plotting
    plt.figure(figsize=(14, 10))
    
    # Group data by feature index
    feature_groups = df.groupby("feature_index")
    
    # Set up data for plotting
    feature_indices = []
    delta_mean_values = []
    delta_mode_values = []
    delta_median_values = []
    delta_mean_last5_values = []
    delta_mode_last5_values = []
    delta_median_last5_values = []
    
    for feature_idx, group in feature_groups:
        feature_indices.append(feature_idx)
        delta_mean_values.append(group["delta_threshold_mean"].mean())
        delta_mode_values.append(group["delta_threshold_mode"].mean())
        delta_median_values.append(group["delta_threshold_median"].mean())
        delta_mean_last5_values.append(group["delta_threshold_mean_last5"].mean())
        delta_mode_last5_values.append(group["delta_threshold_mode_last5"].mean())
        delta_median_last5_values.append(group["delta_threshold_median_last5"].mean())
    
    # Convert to numpy arrays for easier manipulation
    feature_indices = np.array(feature_indices)
    
    # Create subplots
    plt.subplot(2, 1, 1)
    plt.bar(feature_indices - 0.2, delta_mean_values, width=0.2, label='Mean')
    plt.bar(feature_indices, delta_mode_values, width=0.2, label='Mode')
    plt.bar(feature_indices + 0.2, delta_median_values, width=0.2, label='Median')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Feature Index')
    plt.ylabel('Delta Threshold (Pre - Post)')
    plt.title(f'Average Delta Thresholds by Feature Index - All Trials - {condition_name}')
    plt.xticks(feature_indices)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.bar(feature_indices - 0.2, delta_mean_last5_values, width=0.2, label='Mean')
    plt.bar(feature_indices, delta_mode_last5_values, width=0.2, label='Mode')
    plt.bar(feature_indices + 0.2, delta_median_last5_values, width=0.2, label='Median')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Feature Index')
    plt.ylabel('Delta Threshold (Pre - Post)')
    plt.title(f'Average Delta Thresholds by Feature Index - Last 5 Trials - {condition_name}')
    plt.xticks(feature_indices)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(directory_path, f"delta_thresholds_by_feature_{condition_name}.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def create_alpha_correlation_table(directory_path):
    """
    Create a table with angles and threshold values for analysis
    """
    condition_name = os.path.basename(directory_path)
    combined_file = os.path.join(directory_path, f"quest_thresholds_{condition_name}_by_feature.csv")
    
    if not os.path.exists(combined_file):
        print(f"Results file not found: {combined_file}")
        return
    
    # Define the alpha angles corresponding to each feature index
    # Alpha values: 0°, 22.5°, 45°, 67.5°, 90°
    alpha_angles = [0, 22.5, 45, 67.5, 90]
    
    # Load the combined results
    df = pd.read_csv(combined_file)
    
    # Add alpha angle column based on feature index
    df['alpha_angle'] = df['feature_index'].map({i: angle for i, angle in enumerate(alpha_angles)})
    
    # Save the updated dataframe
    output_path = os.path.join(directory_path, f"quest_thresholds_{condition_name}_with_angles.csv")
    df.to_csv(output_path, index=False)
    print(f"Angle correlation table saved to {output_path}")
    
    # Create a summary table by alpha angle
    summary_data = []
    
    for feature_idx, angle in enumerate(alpha_angles):
        feature_data = df[df['feature_index'] == feature_idx]
        
        if len(feature_data) > 0:
            # Calculate mean values for each measure
            summary_row = {
                'alpha_angle': angle,
                'feature_index': feature_idx,
                'participant_count': len(feature_data),
                'avg_delta_threshold_mean': feature_data['delta_threshold_mean'].mean(),
                'avg_delta_threshold_mode': feature_data['delta_threshold_mode'].mean(),
                'avg_delta_threshold_median': feature_data['delta_threshold_median'].mean(),
                'avg_delta_threshold_mean_last5': feature_data['delta_threshold_mean_last5'].mean(),
                'avg_delta_threshold_mode_last5': feature_data['delta_threshold_mode_last5'].mean(),
                'avg_delta_threshold_median_last5': feature_data['delta_threshold_median_last5'].mean()
            }
            summary_data.append(summary_row)
    
    # Create and save summary dataframe
    summary_df = pd.DataFrame(summary_data)
    summary_output_path = os.path.join(directory_path, f"alpha_angle_summary_{condition_name}.csv")
    summary_df.to_csv(summary_output_path, index=False)
    print(f"Alpha angle summary saved to {summary_output_path}")

# Main execution
if __name__ == "__main__":
    # Define the directories to process
    directories = [
        '/Users/robertrutgers/Documents/2Category-Alpha Data/NewAlpha'
        # Add more directories as needed
    ]
    
    for directory_path in directories:
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            continue
        
        # Process the directory
        results = process_directory_with_features(directory_path)
        
        if results is not None:
            # Plot the results
            plot_feature_results(directory_path)
            
            # Create alpha angle correlation table
            create_alpha_correlation_table(directory_path)
            
    print("\nAnalysis complete.")