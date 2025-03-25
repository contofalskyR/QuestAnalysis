import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from psychopy.data import QuestHandler
import os
import math
from scipy import stats 


# Parameters for QuestHandler
startIntensity = 0.1
startSD = 0.3
pThreshold = 0.82
beta = 3.5 
delta = 0.01 
gamma = 0.5


def calculate_mutual_information(alpha_angle, ipl=0.95):
    """
    Calculate mutual information between a feature at a given alpha angle and the category variable.
    """
    # Convert alpha to radians
    alpha_rad = math.radians(alpha_angle)
    
    # For IPL=0.95 (Experiment 3), mutual information has these approximate values:
    # 0° = 0.71 bits
    # 90° = 0 bits
    # Values in between follow approximately a cosine function
    
    # Use cosine function to model the relationship between alpha and mutual information
    if ipl == 0.95:  # Experiment 3
        max_mi = 0.71  # bits at alpha=0
    elif ipl == 0.90:  # Experiment 4
        max_mi = 0.53  # bits at alpha=0
    elif ipl == 0.99:  # Experiment 5
        max_mi = 0.92  # bits at alpha=0
    else:
        max_mi = 0.71  # default to Experiment 3
        
    # Calculate MI using cosine function (matches the data pattern well)
    mi = max_mi * math.cos(alpha_rad)
    
    # Ensure non-negative
    return max(0, mi)

def plot_delta_vs_mutual_information(alpha_thresholds, alpha_angles, group_name, ipl=0.95, suffix=""):
    """
    Plot delta threshold as a function of mutual information (recreating Figure 6).
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
        print("\nComparison to Feldman (2021) Experiment 3:")
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
# Helper function to identify learners based on categorization correctness
def identify_learners(df, correctness_column='correctness', threshold=0.75):
    """
    Identify learners based on their categorization performance.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing participant data
    correctness_column : str
        The column containing categorization correct responses (0 or 1)
    threshold : float
        The minimum accuracy required to be considered a learner
    
    Returns:
    --------
    dict: Dictionary with participant IDs as keys and True (learner) or False (non-learner) as values
    """
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
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
    file_pattern : str
        Pattern to match CSV files
    
    Returns:
    --------
    dict: Results dictionary for the whole sample
    """
    # Fixed settings: mean estimation and last 5 trials
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
        
        # Record discrimination performance if available
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
            
            # Skip if not enough data
            if len(pre_intensities) < 5 or len(pre_responses) < 5 or len(post_intensities) < 5 or len(post_responses) < 5:
                print(f"Skipping feature {feature_idx} for participant {participant}: insufficient data")
                continue
            
            # Extract last 5 trials only
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
    
    """
    Process data and create separate plots for learners and non-learners.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
    file_pattern : str
        Pattern to match CSV files
    estimation_method : str
        Method to estimate threshold: 'mean', 'mode', or 'quantile'
    use_last_five : bool
        Whether to use only the last 5 trials for threshold estimation
    """
    
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
    
    # Create comparison plots
    if learner_results and non_learner_results:
        create_comparison_plots(learner_results, non_learner_results, estimation_method, use_last_five)
    else:
        print("Skipping comparison plots due to missing data for one or more groups")

def process_group(data, participant_list, group_name, estimation_method='mean', use_last_five=True, ipl=0.95):
    """
    Process data for a specific group of participants (learners or non-learners).
    
    Parameters:
    -----------
    data : pandas DataFrame
        Combined data from all participants
    participant_list : list
        List of participant IDs in this group
    group_name : str
        Name of the group for plotting
    estimation_method : str
        Method to estimate threshold: 'mean', 'mode', or 'quantile'
    use_last_five : bool
        Whether to use only the last 5 trials for threshold estimation
        
    Returns:
    --------
    dict: Results dictionary for this group
    """
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
            
            # Skip if not enough data
            if len(pre_intensities) < 5 or len(pre_responses) < 5 or len(post_intensities) < 5 or len(post_responses) < 5:
                print(f"Skipping feature {feature_idx} for participant {participant}: insufficient data")
                continue
            
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
    feature_indices = range(len(alpha_angles))  # 0 to 4 for features 0-4
    feature_labels = [f"Feature{i}\n(α={alpha_angles[i]:.1f}°)" for i in feature_indices]
    
    # Calculate mean delta threshold for each alpha
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
        'Delta threshold is expected to diminish with increasing alpha (angle from max-info dimension).',
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
    
    # Calculate mean pre and post thresholds for each alpha
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
    """
    Run analyses for whole sample and by learning status.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
    ipl : float
        Ideal performance level (default: 0.95 for Experiment 3)
    """
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
    
def calculate_dprime_criterion(hits, false_alarms):
    """
    Calculate d' and criterion measures from hit and false alarm rates.
    """
    # Apply correction for extreme values
    hits = np.clip(hits, 0.01, 0.99)
    false_alarms = np.clip(false_alarms, 0.01, 0.99)
    
    # Calculate d' and criterion
    d_prime = stats.norm.ppf(hits) - stats.norm.ppf(false_alarms)
    criterion = -0.5 * (stats.norm.ppf(hits) + stats.norm.ppf(false_alarms))
    
    return d_prime, criterion


def analyze_sdt_changes(pre_data, post_data, alpha_angles, mutual_info=None):
    """
    Analyze changes in d' and criterion from pre to post training.
    """
    results = {
        'alpha': [],
        'pre_dprime': [],
        'post_dprime': [],
        'delta_dprime': [],
        'pre_criterion': [],
        'post_criterion': [],
        'delta_criterion': [],
        'mutual_info': []
    }
    
    for i, alpha in enumerate(alpha_angles):
        # Extract hit and false alarm rates for this alpha
        pre_hits = pre_data[alpha]['hits']
        pre_fas = pre_data[alpha]['false_alarms']
        post_hits = post_data[alpha]['hits']
        post_fas = post_data[alpha]['false_alarms']
        
        # Calculate pre and post d' and criterion
        pre_d, pre_c = calculate_dprime_criterion(pre_hits, pre_fas)
        post_d, post_c = calculate_dprime_criterion(post_hits, post_fas)
        
        # Calculate delta values
        delta_d = post_d - pre_d
        delta_c = post_c - pre_c
        
        # Store results
        results['alpha'].append(alpha)
        results['pre_dprime'].append(pre_d)
        results['post_dprime'].append(post_d)
        results['delta_dprime'].append(delta_d)
        results['pre_criterion'].append(pre_c)
        results['post_criterion'].append(post_c)
        results['delta_criterion'].append(delta_c)
        
        if mutual_info is not None:
            results['mutual_info'].append(mutual_info[i])
    
    return results


def plot_dprime_mutual_info(results):
    """
    Plot changes in d' against mutual information.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot delta d' vs mutual information
    plt.scatter(results['mutual_info'], results['delta_dprime'], s=80)
    
    # Add labels for each point (alpha angles)
    for i, alpha in enumerate(results['alpha']):
        plt.annotate(f'α={alpha}°', 
                    xy=(results['mutual_info'][i], results['delta_dprime'][i]),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    # Fit regression line if enough points
    if len(results['mutual_info']) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            results['mutual_info'], results['delta_dprime'])
        
        x_line = np.linspace(min(results['mutual_info']), max(results['mutual_info']), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, '--', color='red')
        
        # Add R² text
        plt.text(max(results['mutual_info']) * 0.1, max(results['delta_dprime']) * 0.9,
                f'R² = {r_value**2:.3f}\ny = {slope:.3f}x + {intercept:.3f}',
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add reference line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('Mutual Information (bits)', fontsize=12)
    plt.ylabel('Δd\' (Post - Pre)', fontsize=12)
    plt.title('Change in Perceptual Sensitivity vs Mutual Information', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig("dprime_vs_mi.png", dpi=300, bbox_inches='tight')
    print("D-prime vs MI plot saved to dprime_vs_mi.png")


def analyze_temporal_thresholds(df, participant_list, group_name, alpha_angles, num_segments=3):
    """
    Analyze how discrimination thresholds change over time within the post-test session.
    """
    # Results dictionary
    temporal_results = {angle: {'segment': [], 'threshold': [], 'stderr': []} for angle in alpha_angles}
    
    # Process each alpha angle
    for angle in alpha_angles:
        alpha_data = df[df['participant'].isin(participant_list) & (df['alpha'] == angle)]
        
        # Gather all post-test trials for this alpha
        post_trials = alpha_data[~alpha_data["PostTest.intensity"].isna()]
        
        # Ensure trials are sorted by order
        post_trials = post_trials.sort_values(by=['participant', 'trial_number'])
        
        # For each participant, divide their trials into segments
        all_segment_thresholds = [[] for _ in range(num_segments)]
        
        for participant in participant_list:
            participant_trials = post_trials[post_trials['participant'] == participant]
            
            if len(participant_trials) < num_segments:
                print(f"Skipping participant {participant} for alpha {angle}: insufficient trials")
                continue
                
            # Determine segment sizes
            segment_size = len(participant_trials) // num_segments
            
            for seg in range(num_segments):
                start_idx = seg * segment_size
                end_idx = (seg + 1) * segment_size if seg < num_segments - 1 else len(participant_trials)
                
                segment_trials = participant_trials.iloc[start_idx:end_idx]
                
                # Extract intensities and responses for this segment
                intensities = segment_trials["PostTest.intensity"].dropna().to_list()
                responses = segment_trials["PostTest.response"].dropna().astype(int).to_list()
                
                if len(intensities) < 3 or len(responses) < 3:
                    print(f"Skipping segment {seg+1} for participant {participant}, alpha {angle}: insufficient data")
                    continue
                
                # Create QuestHandler with same parameters as in your code
                quest = QuestHandler(
                    startVal=startIntensity,
                    startValSd=startSD,
                    pThreshold=pThreshold,
                    nTrials=len(intensities),
                    beta=beta,
                    delta=delta,
                    gamma=gamma
                )
                quest.importData(intensities, responses)
                
                # Calculate threshold for this segment
                threshold = quest.mean()
                all_segment_thresholds[seg].append(threshold)
        
        # Calculate average threshold for each segment across participants
        for seg in range(num_segments):
            if all_segment_thresholds[seg]:
                mean_threshold = np.mean(all_segment_thresholds[seg])
                stderr = np.std(all_segment_thresholds[seg], ddof=1) / np.sqrt(len(all_segment_thresholds[seg]))
                
                temporal_results[angle]['segment'].append(seg + 1)
                temporal_results[angle]['threshold'].append(mean_threshold)
                temporal_results[angle]['stderr'].append(stderr)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    for i, angle in enumerate(alpha_angles):
        if temporal_results[angle]['segment']:
            plt.subplot(1, len(alpha_angles), i+1)
            
            plt.errorbar(
                temporal_results[angle]['segment'], 
                temporal_results[angle]['threshold'],
                yerr=temporal_results[angle]['stderr'],
                fmt='o-', 
                label=f'α={angle}°'
            )
            
            plt.xlabel('Time Segment')
            plt.ylabel('Threshold')
            plt.title(f'Alpha = {angle}°')
            plt.grid(alpha=0.3)
    
    plt.suptitle(f'Temporal Dynamics of Post-Test Thresholds - {group_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_filename = f"temporal_thresholds_{group_name.replace(' ', '_')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Temporal threshold analysis saved to {output_filename}")
    
    return temporal_results

# Run the code with hardcoded directory
if __name__ == "__main__":
    data_dir = '/Users/robertrutgers/Documents/2Category-Alpha Data/NewAlpha'
    print(f"Processing data from: {data_dir}")

    # Run the simplified analysis (whole sample + learners/non-learners)
    run_simplified_analyses(data_dir)

# Run the code with hardcoded directory
if __name__ == "__main__":
    data_dir = '/Users/robertrutgers/Documents/2Category-Alpha Data/NewAlpha'
    print(f"Processing data from: {data_dir}")
    
    # Run the analysis with IPL=0.95 (Experiment 3)
    run_simplified_analyses(data_dir, ipl=0.95)
    
    