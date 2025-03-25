import pandas as pd 
import os
import numpy as np
import matplotlib.pyplot as plt
from psychopy.data import QuestHandler 
import scipy.stats as stats 
import math

def calculate_performance_percentage(responses):
    """
    Calculate performance percentage from responses
    
    Parameters:
    -----------
    responses : list
        List of binary responses (0 or 1)
    
    Returns:
    --------
    float
        Percentage of correct responses
    """
    if not responses or len(responses) == 0:
        return None
    return (np.array(responses) == 1).mean() * 100

def calculate_mutual_information(alpha_angle, ipl=0.95):
    """
    Calculate mutual information between a feature 
    at a given alpha angle (feature index)
    and the category variable
    for Experiment 3 (IPL=0.95).
    """
    # Convert alpha to radians
    alpha_rad = math.radians(alpha_angle)
    
    # For Experiment 3, mutual information has max value of 0.71 bits at alpha=0°
    max_mi = 0.71  
        
    # Calculate MI using cosine function (matches the data pattern well)
    mi = max_mi * math.cos(alpha_rad)
    
    # Ensure non-negative
    return max(0, mi)

# Define the discrimination parameters (alpha values)
feature_indices = [0, 1, 2, 3, 4]
alpha_angles = [0, 22.5, 45, 67.5, 90]

# Define the directories for each condition
directory = [
    '/Users/robertrutgers/Documents/2Category-Alpha Data/NewAlpha']

# Parameters for QuestHandler
startIntensity = 0.1
startSD = 0.3
pThreshold = 0.82
beta = 3.5
delta = 0.01 
gamma = 0.5

# Store results across all conditions for plotting
all_results = []

def process_directory(directory_path):
    print(f"\nProcessing directory: {directory_path}")
    
    # Get all CSV files in the directory
    file_list = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    
    if not file_list:
        print(f"No CSV files found in {directory_path}")
        return None
    
    print(f"Found {len(file_list)} CSV files")
    
    # Dictionary to store feature index assignments
    feature_assignments = {}
    
    # Storing results 
    threshold_results = []
    
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        
        try:
            df = pd.read_csv(file_path)
            
            # Check if the required columns exist
            required_columns = ["PreTest.intensity", "PreTest.response", 
                               "PostTest.intensity", "PostTest.response"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Skipping {file_name}: Missing columns {missing_columns}")
                continue
            
            # Get feature index from user if not already assigned
            participant_id = file_name.replace(".csv", "")
            if participant_id not in feature_assignments:
                print(f"Participant: {participant_id}")
                feature_idx = int(input(f"Enter feature index (0-4) for {participant_id}: "))
                
                # Validate input
                if feature_idx not in feature_indices:
                    print(f"Invalid index {feature_idx}. Using 0 as default.")
                    feature_idx = 0
                    
                feature_assignments[participant_id] = feature_idx
            else:
                feature_idx = feature_assignments[participant_id]
                
            # Get corresponding alpha angle and mutual information
            alpha_angle = alpha_angles[feature_idx]
            mutual_info = calculate_mutual_information(alpha_angle)
            
            # Pre & Post-Test Data (all trials)
            pre_intensities = df["PreTest.intensity"].dropna().to_list()
            pre_responses = df["PreTest.response"].dropna().astype(int).to_list()
            post_intensities = df["PostTest.intensity"].dropna().to_list()
            post_responses = df["PostTest.response"].dropna().astype(int).to_list()
            
            # Skip files with no data
            if not pre_intensities or not pre_responses or not post_intensities or not post_responses:
                print(f"Skipping {file_name}: Insufficient data")
                continue
            
            # Last 5 trials data
            pre_intensities_last5 = pre_intensities[-5:] if len(pre_intensities) >= 5 else pre_intensities
            pre_responses_last5 = pre_responses[-5:] if len(pre_responses) >= 5 else pre_responses
            post_intensities_last5 = post_intensities[-5:] if len(post_intensities) >= 5 else post_intensities
            post_responses_last5 = post_responses[-5:] if len(post_responses) >= 5 else post_responses
            
            # Create QuestHandler for all Pre-Test data
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
            
            # Create QuestHandler for all Post-Test data
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
            
            # Create QuestHandler for last 5 Pre-Test trials
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
            
            # Create QuestHandler for last 5 Post-Test trials
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
            
            # Store results 
            threshold_results.append({
                "participant": participant_id,
                "feature_index": feature_idx,
                "alpha_angle": alpha_angle,
                "mutual_information": mutual_info,
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
            
            print(f"Processed {file_name} (Feature {feature_idx}, α={alpha_angle:.1f}°)")
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    if not threshold_results:
        print(f"No valid results found in {directory_path}")
        return None
    
    # Create and save results dataframe
    results_df = pd.DataFrame(threshold_results)
    
    # Get condition name from directory path
    condition_name = os.path.basename(directory_path)
    output_filename = f"quest_thresholds_{condition_name}.csv"
    output_path = os.path.join(directory_path, output_filename)
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    
    # Calculate average delta thresholds across participants
    avg_delta_mean = results_df["delta_threshold_mean"].mean()
    avg_delta_mode = results_df["delta_threshold_mode"].mean()
    avg_delta_median = results_df["delta_threshold_median"].mean()
    
    # Calculate average delta thresholds for last 5 trials
    avg_delta_mean_last5 = results_df["delta_threshold_mean_last5"].mean()
    avg_delta_mode_last5 = results_df["delta_threshold_mode_last5"].mean()
    avg_delta_median_last5 = results_df["delta_threshold_median_last5"].mean()
    
    print(f"\nResults for {condition_name}:")
    print(f"  Participants analyzed: {len(results_df)}")
    print(f"  Average Delta Thresholds (All Trials):")
    print(f"    Mean method: {avg_delta_mean:.4f}")
    print(f"    Mode method: {avg_delta_mode:.4f}")
    print(f"    Median method: {avg_delta_median:.4f}")
    
    print(f"  Average Delta Thresholds (Last 5 Trials):")
    print(f"    Mean method: {avg_delta_mean_last5:.4f}")
    print(f"    Mode method: {avg_delta_mode_last5:.4f}")
    print(f"    Median method: {avg_delta_median_last5:.4f}")
    
    # Print delta thresholds by alpha angle
    print(f"\n  Delta Thresholds by Alpha Angle (Mean Method):")
    
    for feature_idx in sorted(feature_indices):
        feature_data = results_df[results_df['feature_index'] == feature_idx]
        if not feature_data.empty:
            feature_angle = alpha_angles[feature_idx]
            feature_mi = calculate_mutual_information(feature_angle)
            
            mean_delta = feature_data['delta_threshold_mean'].mean()
            mean_delta_last5 = feature_data['delta_threshold_mean_last5'].mean()
            n_participants = len(feature_data)
            
            print(f"    Feature {feature_idx} (α={feature_angle:.1f}°, MI={feature_mi:.3f}): "
                  f"{mean_delta:.4f} (all trials), {mean_delta_last5:.4f} (last 5), n={n_participants}")
    
    # Create plot of delta threshold vs mutual information
    plot_delta_vs_mutual_information(results_df, condition_name, directory_path)
    
    print(f"  Results saved to {output_path}")
    
    # Store results for plotting
    return {
        "condition": condition_name,
        "avg_delta_mean": avg_delta_mean,
        "avg_delta_mode": avg_delta_mode,
        "avg_delta_median": avg_delta_median,
        "avg_delta_mean_last5": avg_delta_mean_last5,
        "avg_delta_mode_last5": avg_delta_mode_last5,
        "avg_delta_median_last5": avg_delta_median_last5,
        "participant_count": len(results_df)
    }

def plot_delta_vs_mutual_information(df, group_name, save_dir="."):
    """
    Plot delta threshold as a function of mutual information.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with processed threshold data
    group_name : str
        Name of the group to include in the plot title
    save_dir : str
        Directory to save the output plot
    """
    plt.figure(figsize=(10, 6))
    
    # Group by feature_index/alpha_angle and calculate mean delta threshold
    grouped_data = df.groupby('feature_index').agg({
        'alpha_angle': 'first',  # Get alpha angle for this feature
        'mutual_information': 'first',  # Get MI for this feature
        'delta_threshold_mean': ['mean', 'std', 'count'],  # Stats for all trials
        'delta_threshold_mean_last5': ['mean', 'std', 'count']  # Stats for last 5
    })
    
    # Extract data for plotting
    alpha_angles = grouped_data['alpha_angle']['first'].values
    mutual_info = grouped_data['mutual_information']['first'].values
    
    delta_means = grouped_data['delta_threshold_mean']['mean'].values
    delta_errors = grouped_data['delta_threshold_mean']['std'].values / np.sqrt(grouped_data['delta_threshold_mean']['count'].values)
    
    delta_means_last5 = grouped_data['delta_threshold_mean_last5']['mean'].values
    delta_errors_last5 = grouped_data['delta_threshold_mean_last5']['std'].values / np.sqrt(grouped_data['delta_threshold_mean_last5']['count'].values)
    
    # Plot for all trials
    plt.errorbar(mutual_info, delta_means, yerr=delta_errors, fmt='o-', 
                color='blue', label='All Trials', linewidth=2, markersize=8, capsize=5)
    
    # Linear regression for all trials
    valid_indices = ~np.isnan(delta_means)
    if sum(valid_indices) > 1:
        x_vals = mutual_info[valid_indices]
        y_vals = delta_means[valid_indices]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        r_squared = r_value**2
        
        # Plot regression line
        x_line = np.linspace(0, max(mutual_info), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, '--', color='blue')
        
        # Add R² text
        plt.text(max(mutual_info) * 0.6, max(delta_means) * 0.9,
                f'All Trials: R² = {r_squared:.3f}\ny = {slope:.3f}x + {intercept:.3f}\np = {p_value:.3f}',
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot for last 5 trials
    plt.errorbar(mutual_info, delta_means_last5, yerr=delta_errors_last5, fmt='s-', 
                color='red', label='Last 5 Trials', linewidth=2, markersize=8, capsize=5)
    
    # Linear regression for last 5 trials
    valid_indices_last5 = ~np.isnan(delta_means_last5)
    if sum(valid_indices_last5) > 1:
        x_vals_last5 = mutual_info[valid_indices_last5]
        y_vals_last5 = delta_means_last5[valid_indices_last5]
        
        # Linear regression
        slope_last5, intercept_last5, r_value_last5, p_value_last5, std_err_last5 = stats.linregress(x_vals_last5, y_vals_last5)
        r_squared_last5 = r_value_last5**2
        
        # Plot regression line
        x_line_last5 = np.linspace(0, max(mutual_info), 100)
        y_line_last5 = slope_last5 * x_line_last5 + intercept_last5
        plt.plot(x_line_last5, y_line_last5, '--', color='red')
        
        # Add R² text for last 5 trials
        plt.text(max(mutual_info) * 0.6, min(delta_means_last5) * 1.1,
                f'Last 5: R² = {r_squared_last5:.3f}\ny = {slope_last5:.3f}x + {intercept_last5:.3f}\np = {p_value_last5:.3f}',
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add annotation for each point (alpha angle)
    for i, (mi, angle) in enumerate(zip(mutual_info, alpha_angles)):
        if not np.isnan(delta_means[i]):
            plt.annotate(f'α={angle:.1f}°', 
                        xy=(mi, delta_means[i]),
                        xytext=(5, 5),
                        textcoords='offset points')
    
    # Set labels and title
    plt.xlabel('Mutual Information (bits)', fontsize=12)
    plt.ylabel('ΔThreshold (Pre - Post)', fontsize=12)
    plt.title(f'ΔThreshold vs Mutual Information - {group_name}', fontsize=14)
    
    # Add reference line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_filename = os.path.join(save_dir, f"delta_vs_mi_{group_name.replace(' ', '_')}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"  Mutual Information plot saved to {output_filename}")

def plot_results(all_results):
    # Create figure for delta thresholds (last 5 trials)
    plt.figure(figsize=(14, 8))
    
    # Extract data for plotting
    conditions = [r["condition"] for r in all_results]
    delta_mean_last5 = [r["avg_delta_mean_last5"] for r in all_results]
    delta_mode_last5 = [r["avg_delta_mode_last5"] for r in all_results]
    delta_median_last5 = [r["avg_delta_median_last5"] for r in all_results]
    
    # Set up bar positions
    x = np.arange(len(conditions))
    width = 0.25
    
    # Create bars
    plt.bar(x - width, delta_mean_last5, width, label='Mean', color='skyblue')
    plt.bar(x, delta_mode_last5, width, label='Mode', color='lightgreen')
    plt.bar(x + width, delta_median_last5, width, label='Median', color='salmon')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Condition')
    plt.ylabel('Delta Threshold (Pre - Post) of Last 5 Trials')
    plt.title('Average Delta Thresholds Across Conditions (Last 5 Trials)')
    plt.xticks(x, conditions, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot to results directory
    output_dir = os.path.dirname(directory[0])
    plt.savefig(os.path.join(output_dir, 'delta_thresholds_last5_comparison.png'))
    print(f"\nPlot saved to {os.path.join(output_dir, 'delta_thresholds_last5_comparison.png')}")
    
    # Create figure for delta thresholds (all trials)
    plt.figure(figsize=(14, 8))
    
    # Extract data for plotting
    delta_mean = [r["avg_delta_mean"] for r in all_results]
    delta_mode = [r["avg_delta_mode"] for r in all_results]
    delta_median = [r["avg_delta_median"] for r in all_results]
    
    # Create bars
    plt.bar(x - width, delta_mean, width, label='Mean', color='skyblue')
    plt.bar(x, delta_mode, width, label='Mode', color='lightgreen')
    plt.bar(x + width, delta_median, width, label='Median', color='salmon')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Condition')
    plt.ylabel('Delta Threshold (Pre - Post)')
    plt.title('Average Delta Thresholds Across Conditions (All Trials)')
    plt.xticks(x, conditions, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot to results directory
    plt.savefig(os.path.join(output_dir, 'delta_thresholds_all_comparison.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'delta_thresholds_all_comparison.png')}")
    
# Add this function to your code
def combine_all_results(directory, output_dir=None):
    """
    Combines all individual condition CSV files into a single consolidated CSV file.
    
    Parameters:
    -----------
    directories : list
        List of directories containing the individual CSV result files
    output_dir : str, optional
        Directory to save the combined CSV file (defaults to parent of first directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(directory[0])
    
    all_data = []
    
    for directory in directory:
        if not os.path.exists(directory):
            continue
            
        condition_name = os.path.basename(directory)
        csv_file = os.path.join(directory, f"quest_thresholds_{condition_name}.csv")
        
        if os.path.exists(csv_file):
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Add a column to identify the condition
                df['condition'] = condition_name
                
                # Append to our combined data
                all_data.append(df)
                
                print(f"Added {len(df)} rows from {condition_name}")
            except Exception as e:
                print(f"Error reading {csv_file}: {str(e)}")
    
    if all_data:
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save combined data
        combined_csv_path = os.path.join(output_dir, "quest_thresholds_all_conditions.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        
        print(f"\nCombined data from all conditions saved to {combined_csv_path}")
        print(f"Total rows: {len(combined_df)}")
        return combined_df
    else:
        print("No data to combine")
        return None

# The rest of your code remains the same...

# Process each directory
results = {}
all_results = []  # Make sure this is defined before use
for directory in directory:
    if os.path.exists(directory):
        print(f"\nProcessing directory: {directory}")
        result = process_directory(directory)
        if result:  # Only add to all_results if we got valid data
            all_results.append(result)
        results[os.path.basename(directory)] = result

# Plot results if we have data from at least one condition
if all_results:
    plot_results(all_results)
else:
    print("No valid results to plot")

print("\nAnalysis complete for all conditions.")
combined_results = combine_all_results(directory)

