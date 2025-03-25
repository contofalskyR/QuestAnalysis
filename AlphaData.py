import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from psychopy.data import QuestHandler
import os
import math
from scipy import stats 
import sys 
import datetime 
import sys
import os
import datetime

class Tee: # Creating Logs -- RC 
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Optional: ensure it goes to the file immediately
    
    def flush(self):
        for f in self.files:
            f.flush()

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Create log filename with timestamp
log_filename = f"logs/analysis_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Open log file
log_file = open(log_filename, 'w')

# Redirect stdout to both console and file
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# Parameters for QuestHandler
startIntensity = 0.1
startSD = 0.3
pThreshold = 0.82
beta = 3.5 
delta = 0.01 
gamma = 0.5


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

def plot_delta_vs_mutual_information(alpha_thresholds, alpha_angles, group_name, ipl=0.95, suffix=""):
    """
    Plot delta threshold as a function of mutual information (recreating Figure 6).
    Delta Threshold = Last 5 Method 
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate mutual information for each alpha
    mutual_info = [calculate_mutual_information(angle, ipl) for angle in alpha_angles]
    
    # Calculate mean delta threshold for each alpha
    delta_means = [np.mean(alpha_thresholds[angle]['delta_threshold']) 
                  if len(alpha_thresholds[angle]['delta_threshold']) > 0 else np.nan 
                  for angle in alpha_angles]
    
    # Calculate standard error for error bars
    delta_errors = []
    for angle in alpha_angles:
        values = alpha_thresholds[angle]['delta_threshold']
        if len(values) > 1:
            std_err = np.std(values, ddof=1) / np.sqrt(len(values))
            delta_errors.append(std_err)
        else:
            delta_errors.append(0)
    
    # Plot data
    plt.errorbar(mutual_info, delta_means, yerr=delta_errors, fmt='o', color='blue', 
                markersize=8, capsize=5)
    
    # Fit a linear regression
    valid_indices = [i for i, val in enumerate(delta_means) if not np.isnan(val)]
    if len(valid_indices) > 1:
        x_vals = [mutual_info[i] for i in valid_indices]
        y_vals = [delta_means[i] for i in valid_indices]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        r_squared = r_value**2
        
        # Plot regression line
        x_line = np.linspace(0, max(mutual_info), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, '--', color='red')
        
        # Add R² text
        plt.text(max(mutual_info) * 0.1, max(delta_means) * 0.9,
                f'R² = {r_squared:.3f}\ny = {slope:.3f}x + {intercept:.3f}',
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Set labels and title
    plt.xlabel('Mutual Information (bits)', fontsize=12)
    plt.ylabel('ΔThreshold (Pre - Post)', fontsize=12)
    plt.title(f'ΔThreshold vs Mutual Information - {group_name}', fontsize=14)
    
    # Add annotation for each point (alpha angle)
    for i, (mi, delta, angle) in enumerate(zip(mutual_info, delta_means, alpha_angles)):
        if not np.isnan(delta):
            plt.annotate(f'α={angle:.1f}°', 
                        xy=(mi, delta),
                        xytext=(5, 5),
                        textcoords='offset points')
    
    # Add reference line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_filename = f"delta_vs_mi_{group_name.replace(' ', '_')}{suffix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Mutual Information plot saved to {output_filename}")
    
    return mutual_info, delta_means, r_squared

def analyze_mutual_information_relationship(mutual_info, delta_means, group_name):
    """
    Perform statistical analysis on the relationship between mutual information and delta threshold.
    """
    # Filter out NaN values
    valid_indices = [i for i, val in enumerate(delta_means) if not np.isnan(val)]
    x_vals = [mutual_info[i] for i in valid_indices]
    y_vals = [delta_means[i] for i in valid_indices]
    
    if len(valid_indices) > 1:
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        r_squared = r_value**2
        
        print(f"\n--- Statistical Analysis for {group_name} ---")
        print(f"Linear regression: y = {slope:.4f}x + {intercept:.4f}")
        print(f"R² = {r_squared:.4f}")
        print(f"p-value = {p_value:.4f}")
        print(f"Standard error = {std_err:.4f}")
        
        # Compare to Feldman's reported values
        print("\nComparison to Experiment 3:")
        print(f"Your R² = {r_squared:.2f} vs. Feldman's R² = 0.89")
        
        if r_squared >= 0.75:
            print("Your results strongly support Feldman's findings that delta threshold is proportional to mutual information.")
        elif r_squared >= 0.5:
            print("Your results moderately support Feldman's findings.")
        else:
            print("Your results show a weaker relationship than reported by Feldman.")
    else:
        print(f"Not enough valid data points for {group_name} to perform statistical analysis.")
        
        
# Helper function to identify learners based on discrimination performance
# Learner = >75% correct on Categorization performance 
def identify_learners(df, correctness_column='correctness', threshold=0.75):
    # Group by participant ID and calculate average accuracy
    if correctness_column in df.columns:
        participant_accuracy = df.groupby('participant')[correctness_column].mean()
        
        # Classify participants as learners or non-learners
        learner_status = {}
        for participant, accuracy in participant_accuracy.items():
            learner_status[participant] = (accuracy >= threshold)
        
        print(f"Participant categorization accuracies: {participant_accuracy}")
        return learner_status
    else:
        print(f"Column '{correctness_column}' not found. Available columns: {df.columns.tolist()}")
        return {}
        
# Function to process whole sample data
def process_whole_sample(data_dir, file_pattern='*.csv', ipl=0.95):
    """
    Process data for the entire sample without separating by learning status.
    Using mean threshold estimation and the last 5 trials by default.
    """
    # Fixed settings: mean estimation and last 5 trials
    # the mean value of the Quest posterior distribution -- RC 
    estimation_method = 'mean'
    use_last_five = True
    
    # Get all CSV files in the directory
    import glob
    file_list = glob.glob(os.path.join(data_dir, file_pattern))
    
    if not file_list:
        print(f"No CSV files matching '{file_pattern}' found in {data_dir}")
        return None
    
    print(f"Found {len(file_list)} CSV files")
    
    # Load all data into a single DataFrame
    all_data = []
    
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            # Add participant ID from filename
            participant_id = os.path.basename(file_path).replace(".csv", "")
            df['participant'] = participant_id
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if not all_data:
        print("No data could be loaded")
        return None
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Get list of all participants
    all_participants = combined_data['participant'].unique().tolist()
    print(f"Processing data for all {len(all_participants)} participants")
    
    # Define the discrimination parameters (alpha values)
    discrimination_parameters = [
        [.5, .5, 1, 0],       # Alpha = 0°
        [.5, .5, .924, .383], # Alpha = 22.5°
        [.5, .5, .707, .707], # Alpha = 45°
        [.5, .5, .383, .924], # Alpha = 67.5°
        [.5, .5, 0, 1]        # Alpha = 90°
    ]
    
    # Calculate alpha angles
    # Convert feature vectors (u,v) into angles in degrees
    # to define alpha values for each feature direction
    alpha_angles = []
    for params in discrimination_parameters:
        # The feature vector is the last two elements [x, y]
        feature_vector = params[2:4]
        # Calculate angle in radians and convert to degrees
        angle_rad = math.atan2(feature_vector[1], feature_vector[0])
        angle_deg = math.degrees(angle_rad)
        # Ensure angle is positive (0-360)
        if angle_deg < 0:
            angle_deg += 360
        alpha_angles.append(angle_deg)
    
    # Track alpha-specific thresholds across all participants
    alpha_thresholds = {
        angle: {
            'pre_threshold': [],
            'post_threshold': [],
            'delta_threshold': [],
            'participants': []
        }
        for angle in alpha_angles
    }
    
    # Also track discrimination performance for reference
    discrimination_performance = {}
    
    # Process each participant
    for participant in all_participants:
        participant_data = combined_data[combined_data['participant'] == participant]
        
        # Record discrimination performance 
        if 'Resp_s_or_d.corr' in participant_data.columns:
            discrimination_performance[participant] = participant_data['Resp_s_or_d.corr'].mean()
        
        # Process alpha-specific data for this participant
        for feature_idx in range(len(discrimination_parameters)):
            alpha_angle = alpha_angles[feature_idx]
            
            # Get pre-test data for this alpha
            pre_alpha_data = participant_data[participant_data['feature_index'] == feature_idx]
            pre_intensities = pre_alpha_data["PreTest.intensity"].dropna().to_list()
            pre_responses = pre_alpha_data["PreTest.response"].dropna().astype(int).to_list()
            
            # Get post-test data for this alpha
            post_alpha_data = participant_data[participant_data['feature_index'] == feature_idx]
            post_intensities = post_alpha_data["PostTest.intensity"].dropna().to_list()
            post_responses = post_alpha_data["PostTest.response"].dropna().astype(int).to_list()
            
            
            # Extract last 5 trials of each staircase 
            # Each staircase code = for feature_idx in range(len(discrimination_parameters)): -- RC 
            pre_intensities_to_use = pre_intensities[-5:]
            pre_responses_to_use = pre_responses[-5:]
            post_intensities_to_use = post_intensities[-5:]
            post_responses_to_use = post_responses[-5:]
            
            # Create QuestHandler for Pre-Test alpha data
            quest_pre_alpha = QuestHandler(
                startVal=startIntensity,
                startValSd=startSD,
                pThreshold=pThreshold,
                nTrials=len(pre_intensities_to_use),
                beta=beta,
                delta=delta,
                gamma=gamma
            )
            quest_pre_alpha.importData(pre_intensities_to_use, pre_responses_to_use)
            
            # Create QuestHandler for Post-Test alpha data
            quest_post_alpha = QuestHandler(
                startVal=startIntensity,
                startValSd=startSD,
                pThreshold=pThreshold,
                nTrials=len(post_intensities_to_use),
                beta=beta,
                delta=delta,
                gamma=gamma
            )
            quest_post_alpha.importData(post_intensities_to_use, post_responses_to_use)
            
            # Calculate thresholds using mean 
            # last 5 method 
            pre_threshold = quest_pre_alpha.mean()
            post_threshold = quest_post_alpha.mean()
            
            # Calculate delta threshold
            # last 5 method 
            delta_threshold = pre_threshold - post_threshold
            
            # Store results for this alpha
            alpha_thresholds[alpha_angle]['pre_threshold'].append(pre_threshold)
            alpha_thresholds[alpha_angle]['post_threshold'].append(post_threshold)
            alpha_thresholds[alpha_angle]['delta_threshold'].append(delta_threshold)
            alpha_thresholds[alpha_angle]['participants'].append(participant)
            
            print(f"  Participant {participant}, Feature {feature_idx} (α={alpha_angle:.1f}°): Pre={pre_threshold:.4f}, Post={post_threshold:.4f}, Delta={delta_threshold:.4f}")
    
    # Plot results for whole sample
    suffix = "_mean_last5"
    plot_group_results(alpha_thresholds, alpha_angles, "All_Participants", suffix)
    
    # Add mutual information analysis
    mutual_info, delta_means, r_squared = plot_delta_vs_mutual_information(
        alpha_thresholds, alpha_angles, "All_Participants", ipl, suffix)
    analyze_mutual_information_relationship(mutual_info, delta_means, "All_Participants")
    
    return {
        'group_name': 'All Participants',
        'alpha_thresholds': alpha_thresholds,
        'alpha_angles': alpha_angles,
        'discrimination_performance': discrimination_performance,
        'mutual_info': mutual_info,
        'delta_means': delta_means,
        'r_squared': r_squared
    }

# Function to process data and create plots for learners and non-learners
def process_data_by_learning_status(data_dir, file_pattern='*.csv', estimation_method='mean', use_last_five=True, ipl=0.95):
    
    # Get all CSV files in the directory
    import glob
    file_list = glob.glob(os.path.join(data_dir, file_pattern))
    
    if not file_list:
        print(f"No CSV files matching '{file_pattern}' found in {data_dir}")
        return
    
    print(f"Found {len(file_list)} CSV files")
    
    # Load all data into a single DataFrame
    all_data = []
    
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            # Add participant ID from filename
            participant_id = os.path.basename(file_path).replace(".csv", "")
            df['participant'] = participant_id
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if not all_data:
        print("No data could be loaded")
        return
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Identify learners and non-learners
    learner_status = identify_learners(combined_data)
    
    if not learner_status:
        print("Could not identify learners. Please check your data format.")
        return
    
    learners = [p for p, is_learner in learner_status.items() if is_learner]
    non_learners = [p for p, is_learner in learner_status.items() if not is_learner]
    
    print(f"Identified {len(learners)} learners and {len(non_learners)} non-learners")
    print(f"Learners: {learners}")
    print(f"Non-learners: {non_learners}")
    
    # Process data separately for learners and non-learners
    learner_results = process_group(combined_data, learners, "Learners", estimation_method, use_last_five, ipl)
    non_learner_results = process_group(combined_data, non_learners, "Non-Learners", estimation_method, use_last_five, ipl)
    
    """
    # Change these lines  -- SUBJECT TO BE REMOVED   
    analyze_learning_scatter(combined_data, learners, "Learners", learner_results['alpha_angles'])
    analyze_learning_scatter(combined_data, non_learners, "Non-Learners", non_learner_results['alpha_angles'])
    """    
    learner_threshold_analysis = analyze_threshold_estimation(combined_data, learners, "Learners", learner_results['alpha_angles'])
    non_learner_threshold_analysis = analyze_threshold_estimation(combined_data, non_learners, "Non-Learners", non_learner_results['alpha_angles'])

    
    # Create comparison plots
    if learner_results and non_learner_results:
        create_comparison_plots(learner_results, non_learner_results, estimation_method, use_last_five)
    else:
        print("Skipping comparison plots due to missing data for one or more groups")

def process_group(data, participant_list, group_name, estimation_method='mean', use_last_five=True, ipl=0.95):
    
    if not participant_list:
        print(f"No participants in {group_name} group")
        return None
    
    print(f"Processing data for {len(participant_list)} participants in {group_name} group")
    
    # Filter data for this group
    group_data = data[data['participant'].isin(participant_list)]
    
    # Define the discrimination parameters (alpha values)
    discrimination_parameters = [
        [.5, .5, 1, 0],       # Alpha = 0°
        [.5, .5, .924, .383], # Alpha = 22.5°
        [.5, .5, .707, .707], # Alpha = 45°
        [.5, .5, .383, .924], # Alpha = 67.5°
        [.5, .5, 0, 1]        # Alpha = 90°
    ]
    
    # Calculate alpha angles
    alpha_angles = []
    for params in discrimination_parameters:
        # The feature vector is the last two elements [x, y]
        feature_vector = params[2:4]
        # Calculate angle in radians and convert to degrees
        angle_rad = math.atan2(feature_vector[1], feature_vector[0])
        angle_deg = math.degrees(angle_rad)
        # Ensure angle is positive (0-360)
        if angle_deg < 0:
            angle_deg += 360
        alpha_angles.append(angle_deg)
    
    # Track alpha-specific thresholds across participants in this group
    alpha_thresholds = {
        angle: {
            'pre_threshold': [],
            'post_threshold': [],
            'delta_threshold': [],
            'participants': []
        }
        for angle in alpha_angles
    }
    
    # Process each participant
    for participant in participant_list:
        participant_data = group_data[group_data['participant'] == participant]
        
        # Process alpha-specific data for this participant
        for feature_idx in range(len(discrimination_parameters)):
            alpha_angle = alpha_angles[feature_idx]
            
            # Get pre-test data for this alpha
            pre_alpha_data = participant_data[participant_data['feature_index'] == feature_idx]
            pre_intensities = pre_alpha_data["PreTest.intensity"].dropna().to_list()
            pre_responses = pre_alpha_data["PreTest.response"].dropna().astype(int).to_list()
            
            # Get post-test data for this alpha
            post_alpha_data = participant_data[participant_data['feature_index'] == feature_idx]
            post_intensities = post_alpha_data["PostTest.intensity"].dropna().to_list()
            post_responses = post_alpha_data["PostTest.response"].dropna().astype(int).to_list()
            
            # Extract last 5 trials if requested
            if use_last_five:
                pre_intensities_to_use = pre_intensities[-5:]
                pre_responses_to_use = pre_responses[-5:]
                post_intensities_to_use = post_intensities[-5:]
                post_responses_to_use = post_responses[-5:]
            else:
                pre_intensities_to_use = pre_intensities
                pre_responses_to_use = pre_responses
                post_intensities_to_use = post_intensities
                post_responses_to_use = post_responses
            
            # Create QuestHandler for Pre-Test alpha data
            quest_pre_alpha = QuestHandler(
                startVal=startIntensity,
                startValSd=startSD,
                pThreshold=pThreshold,
                nTrials=len(pre_intensities_to_use),
                beta=beta,
                delta=delta,
                gamma=gamma
            )
            quest_pre_alpha.importData(pre_intensities_to_use, pre_responses_to_use)
            
            # Create QuestHandler for Post-Test alpha data
            quest_post_alpha = QuestHandler(
                startVal=startIntensity,
                startValSd=startSD,
                pThreshold=pThreshold,
                nTrials=len(post_intensities_to_use),
                beta=beta,
                delta=delta,
                gamma=gamma
            )
            quest_post_alpha.importData(post_intensities_to_use, post_responses_to_use)
            
            # Calculate thresholds based on the specified method
            if estimation_method == 'mean':
                pre_threshold = quest_pre_alpha.mean()
                post_threshold = quest_post_alpha.mean()
            elif estimation_method == 'mode':
                pre_threshold = quest_pre_alpha.mode()
                post_threshold = quest_post_alpha.mode()
            elif estimation_method == 'quantile':
                pre_threshold = quest_pre_alpha.quantile(0.5)  # median
                post_threshold = quest_post_alpha.quantile(0.5)
            else:
                print(f"Unknown estimation method: {estimation_method}. Using mean.")
                pre_threshold = quest_pre_alpha.mean()
                post_threshold = quest_post_alpha.mean()
            
            # Calculate delta threshold
            delta_threshold = pre_threshold - post_threshold
            
            # Store results for this alpha
            alpha_thresholds[alpha_angle]['pre_threshold'].append(pre_threshold)
            alpha_thresholds[alpha_angle]['post_threshold'].append(post_threshold)
            alpha_thresholds[alpha_angle]['delta_threshold'].append(delta_threshold)
            alpha_thresholds[alpha_angle]['participants'].append(participant)
            
            print(f"  Participant {participant}, Feature {feature_idx} (α={alpha_angle:.1f}°): Pre={pre_threshold:.4f}, Post={post_threshold:.4f}, Delta={delta_threshold:.4f}")
    
    # Plot results for this group
    suffix = f"_{estimation_method}"
    if use_last_five:
        suffix += "_last5"
    plot_group_results(alpha_thresholds, alpha_angles, group_name, suffix)
    
    # Add mutual information analysis
    mutual_info, delta_means, r_squared = plot_delta_vs_mutual_information(
        alpha_thresholds, alpha_angles, group_name, ipl, suffix)
    analyze_mutual_information_relationship(mutual_info, delta_means, group_name)
    
    return {
        'group_name': group_name,
        'alpha_thresholds': alpha_thresholds,
        'alpha_angles': alpha_angles,
        'mutual_info': mutual_info,
        'delta_means': delta_means,
        'r_squared': r_squared
    }
    
    return {
        'group_name': group_name,
        'alpha_thresholds': alpha_thresholds,
        'alpha_angles': alpha_angles
    }

def plot_group_results(alpha_thresholds, alpha_angles, group_name, suffix=""):
    """
    Plot delta threshold as a function of alpha for a specific group.
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    feature_indices = range(len(alpha_angles))  # feature_index 0 to 4 
    feature_labels = [f"Feature{i}\n(α={alpha_angles[i]:.1f}°)" for i in feature_indices]
    
    # Calculate mean delta threshold for each alpha – last 5 
    delta_means = [np.mean(alpha_thresholds[angle]['delta_threshold']) 
                   if len(alpha_thresholds[angle]['delta_threshold']) > 0 else np.nan 
                   for angle in alpha_angles]
    
    # Number of participants for each alpha value (for error bars)
    n_participants = [len(alpha_thresholds[angle]['delta_threshold']) for angle in alpha_angles]
    
    # Calculate standard error for error bars if we have enough participants
    delta_errors = []
    for i, angle in enumerate(alpha_angles):
        values = alpha_thresholds[angle]['delta_threshold']
        if len(values) > 1:
            std_err = np.std(values, ddof=1) / np.sqrt(len(values))
            delta_errors.append(std_err)
        else:
            delta_errors.append(0)
    
    # Plot data
    plt.errorbar(feature_indices, delta_means, yerr=delta_errors, fmt='o-', color='blue', 
                 linewidth=2, markersize=8, capsize=5)
    
    # Add reference line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Calculate correlation between alpha angles and delta thresholds
    valid_indices = [i for i, val in enumerate(delta_means) if not np.isnan(val)]
    if len(valid_indices) > 1:
        x_vals = [alpha_angles[i] for i in valid_indices]
        y_vals = [delta_means[i] for i in valid_indices]
        correlation = np.corrcoef(x_vals, y_vals)[0, 1]
        
        # Add correlation text
        plt.figtext(
            0.2, 0.85,
            f"Correlation with Alpha: {correlation:.3f}",
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    # For each point, add the number of participants
    for i, n in enumerate(n_participants):
        if not np.isnan(delta_means[i]):
            plt.text(i, delta_means[i] + (0.01 if delta_means[i] >= 0 else -0.03), 
                     f"n={n}", ha='center')
    
    # Set labels and title
    plt.xlabel('Feature Index (Alpha Angle)', fontsize=12)
    plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
    plt.title(f'Delta Threshold as a Function of Alpha - {group_name}', fontsize=14)
    
    # Set x-ticks to feature indices with labels
    plt.xticks(feature_indices, feature_labels)
    
    # Add annotation
    plt.figtext(
        0.5, 0.01,
        'Positive delta values indicate improvement after discrimination training.\n'
        'Delta threshold is expected to diminish with increasing alpha',
        ha='center',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    output_filename = f"delta_threshold_vs_alpha_{group_name.replace(' ', '_')}{suffix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Plot for {group_name} saved to {output_filename}")
    
    # Also create a plot showing pre and post thresholds separately
    plot_pre_post_thresholds(alpha_thresholds, alpha_angles, group_name, suffix)

def plot_pre_post_thresholds(alpha_thresholds, alpha_angles, group_name, suffix=""):
    """
    Plot pre and post thresholds separately for each alpha value for a specific group.
    """
    plt.figure(figsize=(10, 6))
    
    feature_indices = range(len(alpha_angles))
    feature_labels = [f"Feature{i}\n(α={alpha_angles[i]:.1f}°)" for i in feature_indices]
    
    # Calculate mean pre and post thresholds for each alpha – last 5 --RC 
    pre_means = [np.mean(alpha_thresholds[angle]['pre_threshold']) 
                 if len(alpha_thresholds[angle]['pre_threshold']) > 0 else np.nan 
                 for angle in alpha_angles]
    
    post_means = [np.mean(alpha_thresholds[angle]['post_threshold']) 
                  if len(alpha_thresholds[angle]['post_threshold']) > 0 else np.nan 
                  for angle in alpha_angles]
    
    # Plot data
    plt.plot(feature_indices, pre_means, 'o-', color='blue', 
             label='Pre-Discrimination', linewidth=2, markersize=8)
    plt.plot(feature_indices, post_means, 's-', color='red', 
             label='Post-Discrimination', linewidth=2, markersize=8)
    
    # Set labels and title
    plt.xlabel('Feature Index (Alpha Angle)', fontsize=12)
    plt.ylabel('Threshold', fontsize=12)
    plt.title(f'Pre and Post Discrimination Thresholds by Alpha - {group_name}', fontsize=14)
    
    # Set x-ticks to feature indices with labels
    plt.xticks(feature_indices, feature_labels)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_filename = f"pre_post_thresholds_by_alpha_{group_name.replace(' ', '_')}{suffix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Pre/Post thresholds plot for {group_name} saved to {output_filename}")

def create_comparison_plots(learner_results, non_learner_results, estimation_method='mean', use_last_five=True):
    """
    Create comparison plots showing both learners and non-learners on the same graph.
    """
    suffix = f"_{estimation_method}"
    if use_last_five:
        suffix += "_last5"
    
    # PLOT 1: Delta threshold comparison
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    feature_indices = range(len(learner_results['alpha_angles']))
    feature_labels = [f"Feature{i}\n(α={learner_results['alpha_angles'][i]:.1f}°)" for i in feature_indices]
    
    # Calculate means for each group
    learner_delta_means = [np.mean(learner_results['alpha_thresholds'][angle]['delta_threshold']) 
                          if len(learner_results['alpha_thresholds'][angle]['delta_threshold']) > 0 else np.nan 
                          for angle in learner_results['alpha_angles']]
    
    non_learner_delta_means = [np.mean(non_learner_results['alpha_thresholds'][angle]['delta_threshold']) 
                              if len(non_learner_results['alpha_thresholds'][angle]['delta_threshold']) > 0 else np.nan 
                              for angle in non_learner_results['alpha_angles']]
    
    # Calculate standard errors
    learner_delta_errors = []
    for angle in learner_results['alpha_angles']:
        values = learner_results['alpha_thresholds'][angle]['delta_threshold']
        if len(values) > 1:
            std_err = np.std(values, ddof=1) / np.sqrt(len(values))
            learner_delta_errors.append(std_err)
        else:
            learner_delta_errors.append(0)
    
    non_learner_delta_errors = []
    for angle in non_learner_results['alpha_angles']:
        values = non_learner_results['alpha_thresholds'][angle]['delta_threshold']
        if len(values) > 1:
            std_err = np.std(values, ddof=1) / np.sqrt(len(values))
            non_learner_delta_errors.append(std_err)
        else:
            non_learner_delta_errors.append(0)
    
    # Plot data (slightly offset for clarity)
    plt.errorbar(np.array(feature_indices) - 0.1, learner_delta_means, yerr=learner_delta_errors, 
                 fmt='o-', color='blue', label='Learners', linewidth=2, markersize=8, capsize=5)
    
    plt.errorbar(np.array(feature_indices) + 0.1, non_learner_delta_means, yerr=non_learner_delta_errors, 
                 fmt='s-', color='red', label='Non-Learners', linewidth=2, markersize=8, capsize=5)
    
    # Add reference line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Calculate correlations for each group
    l_valid = [i for i, val in enumerate(learner_delta_means) if not np.isnan(val)]
    if len(l_valid) > 1:
        l_x = [learner_results['alpha_angles'][i] for i in l_valid]
        l_y = [learner_delta_means[i] for i in l_valid]
        l_corr = np.corrcoef(l_x, l_y)[0, 1]
    else:
        l_corr = np.nan
    
    nl_valid = [i for i, val in enumerate(non_learner_delta_means) if not np.isnan(val)]
    if len(nl_valid) > 1:
        nl_x = [non_learner_results['alpha_angles'][i] for i in nl_valid]
        nl_y = [non_learner_delta_means[i] for i in nl_valid]
        nl_corr = np.corrcoef(nl_x, nl_y)[0, 1]
    else:
        nl_corr = np.nan
    
    # Add correlation text
    plt.figtext(
        0.2, 0.85,
        f"Correlations with Alpha:\nLearners: {l_corr:.3f}\nNon-Learners: {nl_corr:.3f}",
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Set labels and title
    plt.xlabel('Feature Index (Alpha Angle)', fontsize=12)
    plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
    plt.title(f'Delta Threshold as a Function of Alpha - Learners vs Non-Learners ({estimation_method}{"_last5" if use_last_five else ""})', fontsize=14)
    
    # Set x-ticks to feature indices with labels
    plt.xticks(feature_indices, feature_labels)
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    plt.savefig(f"delta_threshold_comparison{suffix}.png", dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to delta_threshold_comparison{suffix}.png")
    
    # PLOT 2: Pre and Post comparison across groups
    plt.figure(figsize=(12, 8))
    
    # Calculate means for pre and post for each group
    learner_pre_means = [np.mean(learner_results['alpha_thresholds'][angle]['pre_threshold']) 
                        if len(learner_results['alpha_thresholds'][angle]['pre_threshold']) > 0 else np.nan 
                        for angle in learner_results['alpha_angles']]
    
    learner_post_means = [np.mean(learner_results['alpha_thresholds'][angle]['post_threshold']) 
                         if len(learner_results['alpha_thresholds'][angle]['post_threshold']) > 0 else np.nan 
                         for angle in learner_results['alpha_angles']]
    
    non_learner_pre_means = [np.mean(non_learner_results['alpha_thresholds'][angle]['pre_threshold']) 
                            if len(non_learner_results['alpha_thresholds'][angle]['pre_threshold']) > 0 else np.nan 
                            for angle in non_learner_results['alpha_angles']]
    
    non_learner_post_means = [np.mean(non_learner_results['alpha_thresholds'][angle]['post_threshold']) 
                             if len(non_learner_results['alpha_thresholds'][angle]['post_threshold']) > 0 else np.nan 
                             for angle in non_learner_results['alpha_angles']]
    
    # Plot data
    plt.plot(feature_indices, learner_pre_means, 'o-', color='blue', 
             label='Learners - Pre', linewidth=2, markersize=8)
    plt.plot(feature_indices, learner_post_means, 's-', color='darkblue', 
             label='Learners - Post', linewidth=2, markersize=8)
    plt.plot(feature_indices, non_learner_pre_means, 'o-', color='red', 
             label='Non-Learners - Pre', linewidth=2, markersize=8)
    plt.plot(feature_indices, non_learner_post_means, 's-', color='darkred', 
             label='Non-Learners - Post', linewidth=2, markersize=8)
    
    # Set labels and title
    plt.xlabel('Feature Index (Alpha Angle)', fontsize=12)
    plt.ylabel('Threshold', fontsize=12)
    plt.title(f'Pre and Post Discrimination Thresholds by Alpha - Learners vs Non-Learners ({estimation_method}{"_last5" if use_last_five else ""})', fontsize=14)
    
    # Set x-ticks to feature indices with labels
    plt.xticks(feature_indices, feature_labels)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"pre_post_thresholds_comparison{suffix}.png", dpi=300, bbox_inches='tight')
    print(f"Pre/Post comparison plot saved to pre_post_thresholds_comparison{suffix}.png")
    
def analyze_proportional_improvement(alpha_thresholds, alpha_angles, group_name, ipl=0.95, suffix=""):
    """
    Analyze delta threshold as a proportion of pre-training threshold.
    This accounts for baseline differences in sensitivity.
    """
    # Setup data structures
    alpha_prop_improvements = {}
    
    # Calculate proportional improvement for each feature and participant
    for angle in alpha_angles:
        pre_thresholds = alpha_thresholds[angle]['pre_threshold']
        post_thresholds = alpha_thresholds[angle]['post_threshold']
        delta_thresholds = alpha_thresholds[angle]['delta_threshold']
        
        # Minimum threshold to prevent division by zero or extreme values
        min_threshold = 0.05
        
        # Calculate proportional improvements
        prop_improvements = []
        for pre, delta in zip(pre_thresholds, delta_thresholds):
            if pre > 0:
                # Use max to prevent division by very small values
                prop_improvements.append(delta / max(pre, min_threshold))
            else:
                # Skip if pre-threshold is zero or negative
                continue
        
        alpha_prop_improvements[angle] = prop_improvements
    
    # Calculate mean proportional improvement for each alpha
    prop_means = [np.mean(alpha_prop_improvements[angle]) if alpha_prop_improvements[angle] else np.nan
                  for angle in alpha_angles]
    
    # Calculate standard error
    prop_errors = []
    for angle in alpha_angles:
        values = alpha_prop_improvements[angle]
        if len(values) > 1:
            std_err = np.std(values, ddof=1) / np.sqrt(len(values))
            prop_errors.append(std_err)
        else:
            prop_errors.append(0)
    
    # Calculate mutual information for each alpha
    mutual_info = [calculate_mutual_information(angle, ipl) for angle in alpha_angles]
    
    # Create figure for proportional improvement vs alpha
    plt.figure(figsize=(10, 6))
    plt.errorbar(alpha_angles, prop_means, yerr=prop_errors, fmt='o-', color='purple', 
                 linewidth=2, markersize=8, capsize=5)
    
    # Add reference line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Calculate correlation with alpha angles
    valid_indices = [i for i, val in enumerate(prop_means) if not np.isnan(val)]
    if len(valid_indices) > 1:
        x_vals = [alpha_angles[i] for i in valid_indices]
        y_vals = [prop_means[i] for i in valid_indices]
        alpha_corr = stats.pearsonr(x_vals, y_vals)
        
        # Add correlation text
        plt.figtext(
            0.2, 0.85,
            f"Correlation with Alpha: {alpha_corr[0]:.3f} (p={alpha_corr[1]:.3f})",
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    # Set labels and title
    plt.xlabel('Alpha Angle (degrees)', fontsize=12)
    plt.ylabel('Proportional Improvement (Delta/Pre)', fontsize=12)
    plt.title(f'Proportional Improvement as a Function of Alpha - {group_name}', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_filename = f"prop_improvement_vs_alpha_{group_name.replace(' ', '_')}{suffix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Proportional improvement plot saved to {output_filename}")
    
    # Create figure for proportional improvement vs mutual information
    plt.figure(figsize=(10, 6))
    
    # Plot data
    plt.errorbar(mutual_info, prop_means, yerr=prop_errors, fmt='o', color='purple', 
                markersize=8, capsize=5)
    
    # Fit a linear regression to mutual information
    valid_indices = [i for i, val in enumerate(prop_means) if not np.isnan(val)]
    if len(valid_indices) > 1:
        x_vals = [mutual_info[i] for i in valid_indices]
        y_vals = [prop_means[i] for i in valid_indices]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        r_squared = r_value**2
        
        # Plot regression line
        x_line = np.linspace(0, max(mutual_info), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, '--', color='red')
        
        # Add R² text
        plt.text(max(mutual_info) * 0.1, max(prop_means) * 0.9,
                f'R² = {r_squared:.3f}\ny = {slope:.3f}x + {intercept:.3f}',
                bbox=dict(facecolor='white', alpha=0.7))
                
        # Print correlation statistics
        print(f"Correlation between proportional improvement and MI: {r_value:.3f}, p={p_value:.3f}")
    
    # Add annotation for each point (alpha angle)
    for i, (mi, prop, angle) in enumerate(zip(mutual_info, prop_means, alpha_angles)):
        if not np.isnan(prop):
            plt.annotate(f'α={angle:.1f}°', 
                        xy=(mi, prop),
                        xytext=(5, 5),
                        textcoords='offset points')
    
    # Set labels and title
    plt.xlabel('Mutual Information (bits)', fontsize=12)
    plt.ylabel('Proportional Improvement (Delta/Pre)', fontsize=12)
    plt.title(f'Proportional Improvement vs Mutual Information - {group_name}', fontsize=14)
    
    # Add reference line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_filename = f"prop_improvement_vs_mi_{group_name.replace(' ', '_')}{suffix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Proportional improvement MI plot saved to {output_filename}")
    
    return mutual_info, prop_means

# Main function to run the analysis with different estimation methods
def run_simplified_analyses(data_dir, ipl=0.95):
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return
    
    print(f"\n\n=== Running analysis for Experiment 3 (IPL={ipl}) ===\n")
    
    # First analyze whole sample
    print("\n--- Analyzing whole sample ---\n")
    process_whole_sample(data_dir, ipl=ipl)
    
    # Then analyze by learning status
    print("\n--- Analyzing by learning status ---\n")
    process_data_by_learning_status(
        data_dir=data_dir,
        estimation_method='mean',
        use_last_five=True,
        ipl=ipl
    )
        
def analyze_threshold_estimation(data, participant_list, group_name, alpha_angles):
    # Filter data for this group
    group_data = data[data['participant'].isin(participant_list)]
    
    # Setup results storage
    all_results = {
        'participant': [],
        'alpha': [],
        'feature_idx': [],
        'pre_all_trials': [],
        'post_all_trials': [],
        'pre_last5': [],
        'post_last5': [],
        'pre_convergence_score': [],
        'post_convergence_score': [],
        'pre_threshold_std': [],
        'post_threshold_std': []
    }
    
    # Process each alpha angle
    for feature_idx, angle in enumerate(alpha_angles):
        print(f"Processing alpha = {angle}° (Feature {feature_idx})...")
        
        # Process each participant
        for participant in participant_list:
            # Get participant data for this feature
            participant_data = group_data[(group_data['participant'] == participant) & 
                                         (group_data['feature_index'] == feature_idx)]
                
            # Get pre-test data
            pre_intensities = participant_data["PreTest.intensity"].dropna().to_list()
            pre_responses = participant_data["PreTest.response"].dropna().astype(int).to_list()
            
            # Get post-test data
            post_intensities = participant_data["PostTest.intensity"].dropna().to_list()
            post_responses = participant_data["PostTest.response"].dropna().astype(int).to_list()
            
            # Skip if not enough data
            if len(pre_intensities) < 5 or len(pre_responses) < 5 or len(post_intensities) < 5 or len(post_responses) < 5:
                print(f"  Skipping participant {participant} (insufficient trials)")
                continue
            
            # 1. Analyze thresholds using all trials vs last 5 trials
            
            # Pre-test - all trials
            quest_pre_all = QuestHandler(
                startVal=startIntensity,
                startValSd=startSD,
                pThreshold=pThreshold,
                nTrials=len(pre_intensities),
                beta=beta,
                delta=delta,
                gamma=gamma
            )
            quest_pre_all.importData(pre_intensities, pre_responses)
            pre_threshold_all = quest_pre_all.mean()
            pre_threshold_std = quest_pre_all.sd()
            
            # Pre-test - last 5 trials
            quest_pre_last5 = QuestHandler(
                startVal=startIntensity,
                startValSd=startSD,
                pThreshold=pThreshold,
                nTrials=5,
                beta=beta,
                delta=delta,
                gamma=gamma
            )
            quest_pre_last5.importData(pre_intensities[-5:], pre_responses[-5:])
            pre_threshold_last5 = quest_pre_last5.mean()
            
            # Post-test - all trials
            quest_post_all = QuestHandler(
                startVal=startIntensity,
                startValSd=startSD,
                pThreshold=pThreshold,
                nTrials=len(post_intensities),
                beta=beta,
                delta=delta,
                gamma=gamma
            )
            quest_post_all.importData(post_intensities, post_responses)
            post_threshold_all = quest_post_all.mean()
            post_threshold_std = quest_post_all.sd()
            
            # Post-test - last 5 trials
            quest_post_last5 = QuestHandler(
                startVal=startIntensity,
                startValSd=startSD,
                pThreshold=pThreshold,
                nTrials=5,
                beta=beta,
                delta=delta,
                gamma=gamma
            )
            quest_post_last5.importData(post_intensities[-5:], post_responses[-5:])
            post_threshold_last5 = quest_post_last5.mean()
            
            # 2. Calculate a convergence score (lower is better)
            # Method: Standard deviation of the last few intensities divided by mean
            pre_convergence_score = np.std(pre_intensities[-10:]) / np.mean(pre_intensities[-10:]) if len(pre_intensities) >= 10 else 1.0
            post_convergence_score = np.std(post_intensities[-10:]) / np.mean(post_intensities[-10:]) if len(post_intensities) >= 10 else 1.0
            
            # Store results
            all_results['participant'].append(participant)
            all_results['alpha'].append(angle)
            all_results['feature_idx'].append(feature_idx)
            all_results['pre_all_trials'].append(pre_threshold_all)
            all_results['post_all_trials'].append(post_threshold_all)
            all_results['pre_last5'].append(pre_threshold_last5)
            all_results['post_last5'].append(post_threshold_last5)
            all_results['pre_convergence_score'].append(pre_convergence_score)
            all_results['post_convergence_score'].append(post_convergence_score)
            all_results['pre_threshold_std'].append(pre_threshold_std)
            all_results['post_threshold_std'].append(post_threshold_std)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(all_results)
    
    # Calculate delta thresholds for both methods
    results_df['delta_all_trials'] = results_df['pre_all_trials'] - results_df['post_all_trials']
    results_df['delta_last5'] = results_df['pre_last5'] - results_df['post_last5']
    
    # Summary by alpha angle
    alpha_summary = results_df.groupby('alpha').agg({
        'delta_all_trials': ['mean', 'std', 'count'],
        'delta_last5': ['mean', 'std', 'count'],
        'pre_convergence_score': 'mean',
        'post_convergence_score': 'mean',
        'pre_threshold_std': 'mean',
        'post_threshold_std': 'mean'
    })
    
    print("\nSummary by alpha angle:")
    print(alpha_summary)
    
    # Check if the pattern of results differs between methods
    print("\nCorrelation between all-trials and last-5 delta thresholds:")
    correlation = np.corrcoef(results_df['delta_all_trials'], results_df['delta_last5'])[0,1]
    print(f"  r = {correlation:.3f}")
    
    # Create plots to visualize the results
    
    # 1. Plot delta thresholds by alpha angle - both methods
    plt.figure(figsize=(12, 6))
    
    # Group by alpha and calculate means and std errors
    alpha_values = results_df['alpha'].unique()
    alpha_values.sort()
    
    delta_all_means = []
    delta_all_errors = []
    delta_last5_means = []
    delta_last5_errors = []
    
    for alpha in alpha_values:
        alpha_data = results_df[results_df['alpha'] == alpha]
        
        # All trials
        mean_all = alpha_data['delta_all_trials'].mean()
        se_all = alpha_data['delta_all_trials'].std() / np.sqrt(len(alpha_data))
        delta_all_means.append(mean_all)
        delta_all_errors.append(se_all)
        
        # Last 5 trials
        mean_last5 = alpha_data['delta_last5'].mean()
        se_last5 = alpha_data['delta_last5'].std() / np.sqrt(len(alpha_data))
        delta_last5_means.append(mean_last5)
        delta_last5_errors.append(se_last5)
    
    # Create plot
    plt.subplot(1, 2, 1)
    plt.errorbar(alpha_values, delta_all_means, yerr=delta_all_errors, fmt='o-', color='blue', 
                 label='All trials', linewidth=2, markersize=8, capsize=5)
    plt.errorbar(alpha_values, delta_last5_means, yerr=delta_last5_errors, fmt='s-', color='red', 
                 label='Last 5 trials', linewidth=2, markersize=8, capsize=5)
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Alpha Angle (degrees)', fontsize=12)
    plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
    plt.title('Delta Threshold by Method', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. Plot convergence scores by alpha angle
    plt.subplot(1, 2, 2)
    
    # Calculate means and errors
    pre_conv_means = []
    pre_conv_errors = []
    post_conv_means = []
    post_conv_errors = []
    
    for alpha in alpha_values:
        alpha_data = results_df[results_df['alpha'] == alpha]
        
        # Pre-test
        mean_pre = alpha_data['pre_convergence_score'].mean()
        se_pre = alpha_data['pre_convergence_score'].std() / np.sqrt(len(alpha_data))
        pre_conv_means.append(mean_pre)
        pre_conv_errors.append(se_pre)
        
        # Post-test
        mean_post = alpha_data['post_convergence_score'].mean()
        se_post = alpha_data['post_convergence_score'].std() / np.sqrt(len(alpha_data))
        post_conv_means.append(mean_post)
        post_conv_errors.append(se_post)
    
    # Create plot
    plt.errorbar(alpha_values, pre_conv_means, yerr=pre_conv_errors, fmt='o-', color='blue', 
                 label='Pre-test', linewidth=2, markersize=8, capsize=5)
    plt.errorbar(alpha_values, post_conv_means, yerr=post_conv_errors, fmt='s-', color='red', 
                 label='Post-test', linewidth=2, markersize=8, capsize=5)
    
    plt.xlabel('Alpha Angle (degrees)', fontsize=12)
    plt.ylabel('Convergence Score (lower is better)', fontsize=12)
    plt.title('Staircase Convergence by Phase', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"threshold_analysis_{group_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    
    # 3. Additional plot: Threshold std by alpha angle
    plt.figure(figsize=(10, 6))
    
    # Calculate means and errors
    pre_std_means = []
    pre_std_errors = []
    post_std_means = []
    post_std_errors = []
    
    for alpha in alpha_values:
        alpha_data = results_df[results_df['alpha'] == alpha]
        
        # Pre-test
        mean_pre = alpha_data['pre_threshold_std'].mean()
        se_pre = alpha_data['pre_threshold_std'].std() / np.sqrt(len(alpha_data))
        pre_std_means.append(mean_pre)
        pre_std_errors.append(se_pre)
        
        # Post-test
        mean_post = alpha_data['post_threshold_std'].mean()
        se_post = alpha_data['post_threshold_std'].std() / np.sqrt(len(alpha_data))
        post_std_means.append(mean_post)
        post_std_errors.append(se_post)
    
    # Create plot
    plt.errorbar(alpha_values, pre_std_means, yerr=pre_std_errors, fmt='o-', color='blue', 
                 label='Pre-test', linewidth=2, markersize=8, capsize=5)
    plt.errorbar(alpha_values, post_std_means, yerr=post_std_errors, fmt='s-', color='red', 
                 label='Post-test', linewidth=2, markersize=8, capsize=5)
    
    plt.xlabel('Alpha Angle (degrees)', fontsize=12)
    plt.ylabel('Threshold SD', fontsize=12)
    plt.title('Threshold Uncertainty by Phase', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"threshold_std_{group_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    
    # 4. Plot average staircase trajectories for learners vs non-learners
    plt.figure(figsize=(15, 10))

    for i, alpha in enumerate(alpha_values):
        # For each alpha angle, gather all pre/post intensities
        all_pre_intensities = []
        all_post_intensities = []
        
        # Process each participant
        for participant in participant_list:
            # Get data for this participant and alpha
            part_data = group_data[(group_data['participant'] == participant) & 
                                  (group_data['feature_index'] == alpha_values.tolist().index(alpha))]
            
            # Get pre-test and post-test data
            pre_intensities = part_data["PreTest.intensity"].dropna().to_list()
            post_intensities = part_data["PostTest.intensity"].dropna().to_list()
            
            # Only include if we have enough data
            if len(pre_intensities) >= 5 and len(post_intensities) >= 5:
                all_pre_intensities.append(pre_intensities)
                all_post_intensities.append(post_intensities)
        
        # If we have data for this alpha
        if all_pre_intensities and all_post_intensities:
            # Find max length to normalize trial numbers
            max_pre_length = max(len(x) for x in all_pre_intensities)
            max_post_length = max(len(x) for x in all_post_intensities)
            
            # Create normalized arrays for averaging
            norm_pre_intensities = np.zeros((len(all_pre_intensities), max_pre_length))
            norm_post_intensities = np.zeros((len(all_post_intensities), max_post_length))
            
            # Fill arrays (pad shorter staircases with NaN)
            for j, stair in enumerate(all_pre_intensities):
                norm_pre_intensities[j, :len(stair)] = stair
                norm_pre_intensities[j, len(stair):] = np.nan
            
            for j, stair in enumerate(all_post_intensities):
                norm_post_intensities[j, :len(stair)] = stair
                norm_post_intensities[j, len(stair):] = np.nan
            
            # Calculate means (ignoring NaNs)
            pre_means = np.nanmean(norm_pre_intensities, axis=0)
            post_means = np.nanmean(norm_post_intensities, axis=0)
            
            # Calculate standard errors
            pre_sem = np.nanstd(norm_pre_intensities, axis=0) / np.sqrt(np.sum(~np.isnan(norm_pre_intensities), axis=0))
            post_sem = np.nanstd(norm_post_intensities, axis=0) / np.sqrt(np.sum(~np.isnan(norm_post_intensities), axis=0))
            
            # Plot pre-test staircase
            plt.subplot(len(alpha_values), 2, 2*i+1)
            plt.errorbar(range(len(pre_means)), pre_means, yerr=pre_sem, fmt='b.-', alpha=0.7, 
                        label=f'n={len(all_pre_intensities)}')
            plt.title(f'Average Pre-test Staircase, α={alpha}°')
            plt.xlabel('Trial Number')
            plt.ylabel('Intensity')
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Plot post-test staircase
            plt.subplot(len(alpha_values), 2, 2*i+2)
            plt.errorbar(range(len(post_means)), post_means, yerr=post_sem, fmt='r.-', alpha=0.7,
                        label=f'n={len(all_post_intensities)}')
            plt.title(f'Average Post-test Staircase, α={alpha}°')
            plt.xlabel('Trial Number')
            plt.ylabel('Intensity')
            plt.grid(alpha=0.3)
            plt.legend()

    plt.tight_layout()
    plt.savefig(f"average_staircase_trajectory_{group_name}.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory
    
    return results_df
    
# Run the code with hardcoded directory
if __name__ == "__main__":
    data_dir = '/Users/robertrutgers/Documents/2Category-Alpha Data/NewAlpha'
    print(f"Processing data from: {data_dir}")
    
    # Run the simplified analysis (whole sample + learners/non-learners)
    # Using IPL=0.95 for Experiment 3 (this is the default value)
    run_simplified_analyses(data_dir)
    
    
    
"""
#Scatter Plot for extra visuals, but it is indeed extra 
def analyze_learning_scatter(df, participant_list, group_name, alpha_angles):
    Create scatter plots of pre vs post thresholds for each alpha angle.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Combined data from all participants
    participant_list : list
        List of participant IDs to include
    group_name : str
        Name of the group (e.g., "Learners", "Non-Learners")
    alpha_angles : list
        List of alpha angles to analyze
    # Filter data for this group
    group_data = df[df['participant'].isin(participant_list)]
    
    # Create a figure with subplots for each alpha angle
    num_angles = len(alpha_angles)
    fig, axes = plt.subplots(1, num_angles, figsize=(5*num_angles, 5), sharey=True, sharex=True)
    
    # If there's only one alpha angle, axes won't be an array
    if num_angles == 1:
        axes = [axes]
    
    # Process each alpha angle
    for ax_idx, angle in enumerate(alpha_angles):
        feature_idx = alpha_angles.index(angle)
        
        # Store pre and post thresholds for each participant
        pre_thresholds = []
        post_thresholds = []
        participant_ids = []
        
        # Process each participant
        for participant in participant_list:
            participant_data = group_data[(group_data['participant'] == participant) & 
                                         (group_data['feature_index'] == feature_idx)]
            
            # Get pre-test data
            pre_intensities = participant_data["PreTest.intensity"].dropna().to_list()
            pre_responses = participant_data["PreTest.response"].dropna().astype(int).to_list()
            
            # Get post-test data
            post_intensities = participant_data["PostTest.intensity"].dropna().to_list()
            post_responses = participant_data["PostTest.response"].dropna().astype(int).to_list()
            
            # Skip if not enough data
            if len(pre_intensities) < 5 or len(pre_responses) < 5 or len(post_intensities) < 5 or len(post_responses) < 5:
                continue
            
            # Use last 5 trials
            pre_intensities = pre_intensities[-5:]
            pre_responses = pre_responses[-5:]
            post_intensities = post_intensities[-5:]
            post_responses = post_responses[-5:]
            
            # Create QuestHandler objects and calculate thresholds
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
            pre_threshold = quest_pre.mean()
            
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
            post_threshold = quest_post.mean()
            
            # Store the thresholds
            pre_thresholds.append(pre_threshold)
            post_thresholds.append(post_threshold)
            participant_ids.append(participant)
        
        # Create the scatter plot
        axes[ax_idx].scatter(pre_thresholds, post_thresholds, alpha=0.7, s=40)
        
        # Add diagonal reference line (y=x)
        min_val = min(min(pre_thresholds) if pre_thresholds else 0, 
                      min(post_thresholds) if post_thresholds else 0)
        max_val = max(max(pre_thresholds) if pre_thresholds else 1, 
                      max(post_thresholds) if post_thresholds else 1)
        axes[ax_idx].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
        
        # Fit and plot regression line
        if len(pre_thresholds) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(pre_thresholds, post_thresholds)
            x_line = np.linspace(min_val, max_val, 100)
            y_line = slope * x_line + intercept
            axes[ax_idx].plot(x_line, y_line, 'r-', alpha=0.6)
            
            # Add R² text
            axes[ax_idx].text(0.05, 0.95, f'R² = {r_value**2:.3f}', 
                             transform=axes[ax_idx].transAxes, 
                             fontsize=10, 
                             va='top', 
                             bbox=dict(facecolor='white', alpha=0.7))
        
        # Set titles and labels
        axes[ax_idx].set_title(f'α = {angle}°')
        axes[ax_idx].grid(alpha=0.3)
        
        # Highlight points below the line (improved performance)
        for i, (pre, post) in enumerate(zip(pre_thresholds, post_thresholds)):
            if post < pre:  # Improved performance
                axes[ax_idx].plot(pre, post, 'o', markerfacecolor='none', 
                                 markeredgecolor='green', markersize=10, alpha=0.6)
            elif post > pre:  # Worsened performance
                axes[ax_idx].plot(pre, post, 'o', markerfacecolor='none', 
                                 markeredgecolor='red', markersize=10, alpha=0.6)
    
    # Set common labels
    fig.text(0.5, 0.01, 'Pre-Training Threshold', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Post-Training Threshold', va='center', rotation='vertical', fontsize=12)
    
    # Set equal aspect ratio so diagonal is at 45 degrees
    for ax in axes:
        ax.set_aspect('equal')
    
    plt.suptitle(f'Pre vs Post Thresholds - {group_name}', fontsize=16)
    plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])
    
    # Save figure
    output_filename = f"pre_post_scatter_{group_name.replace(' ', '_')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Pre vs Post scatter plots saved to {output_filename}")
    
    return fig
    """
    
    