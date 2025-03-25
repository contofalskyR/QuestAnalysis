import pandas as pd 
import os
import numpy as np
import matplotlib.pyplot as plt
from psychopy.data import QuestHandler 
import scipy.stats as stats 

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

# Define the directories for each condition
directory = [
    '/Users/robertrutgers/Documents/2Category-Alpha Data']

# Parameters for QuestHandler
startIntensity = 0.1
startSD = 0.3
pThreshold = 0.82
beta = 1 
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
            participant_id = file_name.replace(".csv", "")
            threshold_results.append({
                "participant": participant_id,
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
        
def compare_all_raw_vs_quest(directory):
    """
    Compare Raw vs Quest thresholds across all conditions
    
    Parameters:
    -----------
    directories : list
        List of directories containing the CSV data files
        
    Returns:
    --------
    None (creates and saves plots)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from psychopy.data import QuestHandler
    import os
    
    # Parameters for QuestHandler (same as in main code)
    startIntensity = 0.1
    startSD = 0.3
    pThreshold = 0.82
    beta = 1 
    delta = 0.01 
    gamma = 0.5
    
    # Data collection for all conditions
    all_conditions_data = []
    
    # Process each directory
    for directory in directory:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
            
        condition_name = os.path.basename(directory)
        print(f"\nProcessing {condition_name} for raw vs quest comparison...")
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') and not f.startswith('quest_thresholds_')]
        
        condition_raw_pre_means = []
        condition_raw_post_means = []
        condition_quest_pre_means = []
        condition_quest_post_means = []
        condition_quest_pre_modes = []
        condition_quest_post_modes = []
        condition_quest_pre_medians = []
        condition_quest_post_medians = []
        
        for file_name in csv_files:
            file_path = os.path.join(directory, file_name)
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Prepare data
                pre_intensities = df["PreTest.intensity"].dropna().tolist()
                post_intensities = df["PostTest.intensity"].dropna().tolist()
                pre_responses = df["PreTest.response"].dropna().astype(int).tolist()
                post_responses = df["PostTest.response"].dropna().astype(int).tolist()
                
                # Skip if not enough data
                if not pre_intensities or not pre_responses or not post_intensities or not post_responses:
                    continue
                
                # Raw means
                pre_raw_mean = np.mean(pre_intensities)
                post_raw_mean = np.mean(post_intensities)
                
                # Create QuestHandler for pre-test
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
                
                # Create QuestHandler for post-test
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
                
                # Quest estimates
                pre_quest_mean = quest_pre.mean()
                post_quest_mean = quest_post.mean()
                pre_quest_mode = quest_pre.mode()
                post_quest_mode = quest_post.mode()
                pre_quest_median = quest_pre.quantile(0.5)
                post_quest_median = quest_post.quantile(0.5)
                
                # Store values
                condition_raw_pre_means.append(pre_raw_mean)
                condition_raw_post_means.append(post_raw_mean)
                condition_quest_pre_means.append(pre_quest_mean)
                condition_quest_post_means.append(post_quest_mean)
                condition_quest_pre_modes.append(pre_quest_mode)
                condition_quest_post_modes.append(post_quest_mode)
                condition_quest_pre_medians.append(pre_quest_median)
                condition_quest_post_medians.append(post_quest_median)
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
        
        # Calculate averages for the condition
        if condition_raw_pre_means:  # Only if we have data
            condition_data = {
                'condition': condition_name,
                'raw_pre_mean': np.mean(condition_raw_pre_means),
                'raw_post_mean': np.mean(condition_raw_post_means),
                'raw_delta_mean': np.mean(condition_raw_pre_means) - np.mean(condition_raw_post_means),
                'quest_pre_mean': np.mean(condition_quest_pre_means),
                'quest_post_mean': np.mean(condition_quest_post_means),
                'quest_delta_mean': np.mean(condition_quest_pre_means) - np.mean(condition_quest_post_means),
                'quest_pre_mode': np.mean(condition_quest_pre_modes),
                'quest_post_mode': np.mean(condition_quest_post_modes),
                'quest_delta_mode': np.mean(condition_quest_pre_modes) - np.mean(condition_quest_post_modes),
                'quest_pre_median': np.mean(condition_quest_pre_medians),
                'quest_post_median': np.mean(condition_quest_post_medians),
                'quest_delta_median': np.mean(condition_quest_pre_medians) - np.mean(condition_quest_post_medians),
                'n_participants': len(condition_raw_pre_means)
            }
            
            all_conditions_data.append(condition_data)
            print(f"  Added data for {condition_name} with {condition_data['n_participants']} participants")
    
    # Convert to DataFrame for easier handling
    results_df = pd.DataFrame(all_conditions_data)
    
    # Save comprehensive results
    output_dir = os.path.dirname(directory[0])
    results_df.to_csv(os.path.join(output_dir, 'raw_vs_quest_all_conditions.csv'), index=False)
    
    if len(results_df) == 0:
        print("No data to plot")
        return
    
    # Plotting
    # 1. RAW vs QUEST Pre-Test values across conditions
    plt.figure(figsize=(16, 10))
    
    # Extract data
    conditions = results_df['condition'].tolist()
    raw_pre = results_df['raw_pre_mean'].tolist()
    quest_pre_mean = results_df['quest_pre_mean'].tolist()
    quest_pre_mode = results_df['quest_pre_mode'].tolist()
    quest_pre_median = results_df['quest_pre_median'].tolist()
    
    # Set up positions
    x = np.arange(len(conditions))
    width = 0.2
    
    # Create bars for Pre-Test
    plt.subplot(2, 1, 1)
    plt.bar(x - 1.5*width, raw_pre, width, label='Raw Mean', color='skyblue')
    plt.bar(x - 0.5*width, quest_pre_mean, width, label='Quest Mean', color='lightgreen')
    plt.bar(x + 0.5*width, quest_pre_mode, width, label='Quest Mode', color='salmon')
    plt.bar(x + 1.5*width, quest_pre_median, width, label='Quest Median', color='purple')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Pre-Test Threshold')
    plt.title('Raw vs Quest Pre-Test Thresholds Across Conditions')
    plt.xticks(x, conditions, rotation=45, ha='right')
    plt.legend()
    
    # Extract data for Post-Test
    raw_post = results_df['raw_post_mean'].tolist()
    quest_post_mean = results_df['quest_post_mean'].tolist()
    quest_post_mode = results_df['quest_post_mode'].tolist()
    quest_post_median = results_df['quest_post_median'].tolist()
    
    # Create bars for Post-Test
    plt.subplot(2, 1, 2)
    plt.bar(x - 1.5*width, raw_post, width, label='Raw Mean', color='skyblue')
    plt.bar(x - 0.5*width, quest_post_mean, width, label='Quest Mean', color='lightgreen')
    plt.bar(x + 0.5*width, quest_post_mode, width, label='Quest Mode', color='salmon')
    plt.bar(x + 1.5*width, quest_post_median, width, label='Quest Median', color='purple')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Condition')
    plt.ylabel('Post-Test Threshold')
    plt.title('Raw vs Quest Post-Test Thresholds Across Conditions')
    plt.xticks(x, conditions, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'raw_vs_quest_pre_post_comparison.png'), dpi=300)
    
    # 2. RAW vs QUEST Delta values
    plt.figure(figsize=(14, 8))
    
    # Extract delta data
    raw_delta = results_df['raw_delta_mean'].tolist()
    quest_delta_mean = results_df['quest_delta_mean'].tolist()
    quest_delta_mode = results_df['quest_delta_mode'].tolist()
    quest_delta_median = results_df['quest_delta_median'].tolist()
    
    # Create bars for Deltas
    plt.bar(x - 1.5*width, raw_delta, width, label='Raw Mean Delta', color='skyblue')
    plt.bar(x - 0.5*width, quest_delta_mean, width, label='Quest Mean Delta', color='lightgreen')
    plt.bar(x + 0.5*width, quest_delta_mode, width, label='Quest Mode Delta', color='salmon')
    plt.bar(x + 1.5*width, quest_delta_median, width, label='Quest Median Delta', color='purple')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Condition')
    plt.ylabel('Delta Threshold (Pre - Post)')
    plt.title('Raw vs Quest Delta Thresholds Across Conditions')
    plt.xticks(x, conditions, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'raw_vs_quest_delta_comparison.png'), dpi=300)
    
    print(f"\nPlots saved to {output_dir}")
    print("1. raw_vs_quest_pre_post_comparison.png")
    print("2. raw_vs_quest_delta_comparison.png")
    
    # Print summary table
    print("\nSummary of Results:")
    print(results_df[['condition', 'n_participants', 'raw_delta_mean', 
                     'quest_delta_mean', 'quest_delta_mode', 'quest_delta_median']])
    
    return results_df

# Process each directory
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

# Add this line to create the new comparison
print("\nComparing Raw vs Quest thresholds across all conditions...")
compare_all_raw_vs_quest(directory)

print("\nAnalysis complete for all conditions.")
combined_results = combine_all_results(directory)

# Learner-specific Analysis
def learner_quest_analysis(directory, performance_threshold=75):
    """
    Perform Quest analysis specifically for learners
    """
    learner_results = {}
    
    for directory in directory:
        condition_name = os.path.basename(directory)
        condition_learners = []
        
        # Get CSV files in the directory
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') 
                     and not f.startswith('quest_') 
                     and not f.startswith('raw_')]
        
        for file_name in csv_files:
            file_path = os.path.join(directory, file_name)
            
            try:
                df = pd.read_csv(file_path)
                
                # Check for required columns
                if not all(col in df.columns for col in ["PreTest.intensity", "PreTest.response", 
                                                         "PostTest.intensity", "PostTest.response"]):
                    continue
                
                # Calculate performance
                pre_responses = df["PreTest.response"].dropna().astype(int).tolist()
                post_responses = df["PostTest.response"].dropna().astype(int).tolist()
                
                pre_performance = calculate_performance_percentage(pre_responses)
                post_performance = calculate_performance_percentage(post_responses)
                
                # Check if participant is a learner
                if (pre_performance is not None and 
                    post_performance is not None and 
                    pre_performance > performance_threshold and 
                    post_performance > performance_threshold):
                    
                    # Collect learner data
                    learner_data = {
                        'file_name': file_name,
                        'pre_performance': pre_performance,
                        'post_performance': post_performance,
                        'pre_intensities': df["PreTest.intensity"].dropna().tolist(),
                        'pre_responses': pre_responses,
                        'post_intensities': df["PostTest.intensity"].dropna().tolist(),
                        'post_responses': post_responses
                    }
                    condition_learners.append(learner_data)
            
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
        
        # Perform Quest analysis on learners
        if condition_learners:
            try:
                learner_pre_intensities = [item for learner in condition_learners for item in learner['pre_intensities']]
                learner_pre_responses = [item for learner in condition_learners for item in learner['pre_responses']]
                learner_post_intensities = [item for learner in condition_learners for item in learner['post_intensities']]
                learner_post_responses = [item for learner in condition_learners for item in learner['post_responses']]
                
                # Ensure we have enough unique data points
                if len(set(learner_pre_intensities)) > 1 and len(set(learner_pre_responses)) > 1:
                    # Create Quest Handlers for learners
                    quest_pre_learners = QuestHandler(
                        startVal=startIntensity,
                        startValSd=startSD,
                        pThreshold=pThreshold,
                        nTrials=len(learner_pre_intensities),
                        beta=beta,
                        delta=delta,
                        gamma=gamma
                    )
                    quest_pre_learners.importData(learner_pre_intensities, learner_pre_responses)
                    
                    quest_post_learners = QuestHandler(
                        startVal=startIntensity,
                        startValSd=startSD,
                        pThreshold=pThreshold,
                        nTrials=len(learner_post_intensities),
                        beta=beta,
                        delta=delta,
                        gamma=gamma
                    )
                    quest_post_learners.importData(learner_post_intensities, learner_post_responses)
                    
                    # Calculate learner-specific thresholds
                    learner_results[condition_name] = {
                        'n_learners': len(condition_learners),
                        'pre_threshold_mean': quest_pre_learners.mean(),
                        'post_threshold_mean': quest_post_learners.mean(),
                        'delta_threshold_mean': quest_pre_learners.mean() - quest_post_learners.mean(),
                        'pre_threshold_mode': quest_pre_learners.mode(),
                        'post_threshold_mode': quest_post_learners.mode(),
                        'delta_threshold_mode': quest_pre_learners.mode() - quest_post_learners.mode(),
                        'learner_details': condition_learners
                    }
                else:
                    print(f"Not enough unique data points for learners in {condition_name}")
            
            except Exception as e:
                print(f"Error in Quest analysis for learners in {condition_name}: {str(e)}")
    
    # Print and return results
    print("\nLearner-specific Quest Analysis:")
    for condition, results in learner_results.items():
        print(f"\n{condition}:")
        print(f"  Number of Learners: {results['n_learners']}")
        print(f"  Mean Threshold Delta: {results['delta_threshold_mean']:.4f}")
        print(f"  Mode Threshold Delta: {results['delta_threshold_mode']:.4f}")
    
    return learner_results

# Run learner-specific Quest analysis
learner_quest_results = learner_quest_analysis(directory)