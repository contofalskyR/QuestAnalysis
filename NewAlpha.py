import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from psychopy.data import QuestHandler
import os
import math

# Parameters for QuestHandler
startIntensity = 0.1
startSD = 0.3
pThreshold = 0.82
beta = 1 
delta = 0.01 
gamma = 0.5

# Helper function to identify learners based on discrimination performance
def identify_learners(df, discrimination_column='Resp_s_or_d.corr', threshold=0.75):
    """
    Identify learners based on their discrimination performance.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing participant data
    discrimination_column : str
        The column containing discrimination correct responses (0 or 1)
    threshold : float
        The minimum accuracy required to be considered a learner
    
    Returns:
    --------
    dict: Dictionary with participant IDs as keys and True (learner) or False (non-learner) as values
    """
    # Group by participant ID and calculate average accuracy
    if discrimination_column in df.columns:
        participant_accuracy = df.groupby('participant')[discrimination_column].mean()
        
        # Classify participants as learners or non-learners
        learner_status = {}
        for participant, accuracy in participant_accuracy.items():
            learner_status[participant] = (accuracy >= threshold)
        
        print(f"Participant accuracies: {participant_accuracy}")
        return learner_status
    else:
        print(f"Column '{discrimination_column}' not found. Available columns: {df.columns.tolist()}")
        return {}

# Function to process whole sample data
def process_whole_sample(data_dir, file_pattern='*.csv', estimation_method='mean', use_last_five=True):
    """
    Process data for the entire sample without separating by learning status.
    Using mean threshold estimation and the last 5 trials by default.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
    file_pattern : str
        Pattern to match CSV files
    estimation_method : str
        Method to estimate threshold (fixed to 'mean')
    use_last_five : bool
        Whether to use only the last 5 trials for threshold estimation (fixed to True)
    
    Returns:
    --------
    dict: Results dictionary for the whole sample
    """
    # Override parameters to ensure we're using mean and last 5 trials
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
            
            # Calculate thresholds based on mean estimation method
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

def plot_whole_sample_results(alpha_thresholds, alpha_angles, discrimination_performance=None, suffix="_mean_last5"):
    """
    Plot delta threshold as a function of alpha for the whole sample.
    
    Parameters:
    -----------
    alpha_thresholds : dict
        Dictionary containing threshold data for each alpha angle
    alpha_angles : list
        List of alpha angles in degrees
    discrimination_performance : dict, optional
        Dictionary of discrimination performance by participant
    suffix : str
        Suffix for output filename (defaults to "_mean_last5")
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
    plt.errorbar(feature_indices, delta_means, yerr=delta_errors, fmt='o-', color='purple', 
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
    plt.title(f'Delta Threshold as a Function of Alpha - All Participants', fontsize=14)
    
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
    
    # If we have discrimination performance data, add correlation with delta
    if discrimination_performance:
        # Calculate average delta threshold for each participant
        participant_deltas = {}
        for angle in alpha_angles:
            for i, participant in enumerate(alpha_thresholds[angle]['participants']):
                if participant not in participant_deltas:
                    participant_deltas[participant] = []
                participant_deltas[participant].append(alpha_thresholds[angle]['delta_threshold'][i])
        
        avg_participant_deltas = {p: np.mean(deltas) for p, deltas in participant_deltas.items()}
        
        # Calculate correlation between discrimination performance and average delta
        common_participants = set(discrimination_performance.keys()) & set(avg_participant_deltas.keys())
        if len(common_participants) > 1:
            discrim_values = [discrimination_performance[p] for p in common_participants]
            delta_values = [avg_participant_deltas[p] for p in common_participants]
            perf_corr = np.corrcoef(discrim_values, delta_values)[0, 1]
            
            # Add this information to the plot
            plt.figtext(
                0.2, 0.75,
                f"Correlation with Discrimination Performance: {perf_corr:.3f}\n(n={len(common_participants)})",
                bbox=dict(facecolor='white', alpha=0.7)
            )
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    output_filename = f"delta_threshold_vs_alpha_all_participants{suffix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Plot for all participants saved to {output_filename}")
    
    # Also create a plot showing pre and post thresholds separately
    plot_whole_sample_pre_post(alpha_thresholds, alpha_angles, suffix)
    
    # Create individual participant plots if there are at least 5 participants
    total_participants = len(set().union(*[set(alpha_thresholds[angle]['participants']) for angle in alpha_angles]))
    if total_participants >= 5:
        plot_individual_participants(alpha_thresholds, alpha_angles, suffix)

def plot_whole_sample_pre_post(alpha_thresholds, alpha_angles, suffix="_mean_last5"):
    """
    Plot pre and post thresholds separately for each alpha value for the whole sample.
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
    
    # Calculate standard errors
    pre_errors = []
    post_errors = []
    for angle in alpha_angles:
        pre_values = alpha_thresholds[angle]['pre_threshold']
        if len(pre_values) > 1:
            pre_err = np.std(pre_values, ddof=1) / np.sqrt(len(pre_values))
            pre_errors.append(pre_err)
        else:
            pre_errors.append(0)
            
        post_values = alpha_thresholds[angle]['post_threshold']
        if len(post_values) > 1:
            post_err = np.std(post_values, ddof=1) / np.sqrt(len(post_values))
            post_errors.append(post_err)
        else:
            post_errors.append(0)
    
    # Plot data with error bars
    plt.errorbar(feature_indices, pre_means, yerr=pre_errors, fmt='o-', color='blue', 
                 label='Pre-Discrimination', linewidth=2, markersize=8, capsize=5)
    plt.errorbar(feature_indices, post_means, yerr=post_errors, fmt='s-', color='red', 
                 label='Post-Discrimination', linewidth=2, markersize=8, capsize=5)
    
    # Set labels and title
    plt.xlabel('Feature Index (Alpha Angle)', fontsize=12)
    plt.ylabel('Threshold', fontsize=12)
    plt.title(f'Pre and Post Discrimination Thresholds by Alpha - All Participants', fontsize=14)
    
    # Set x-ticks to feature indices with labels
    plt.xticks(feature_indices, feature_labels)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_filename = f"pre_post_thresholds_by_alpha_all_participants{suffix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Pre/Post thresholds plot for all participants saved to {output_filename}")

def plot_individual_participants(alpha_thresholds, alpha_angles, suffix="_mean_last5"):
    """
    Create a scatter plot showing individual participant delta thresholds 
    as a function of alpha angle, color-coded by participant.
    """
    plt.figure(figsize=(12, 8))
    
    # Get unique list of all participants across all alpha angles
    all_participants = set()
    for angle in alpha_angles:
        all_participants.update(alpha_thresholds[angle]['participants'])
    all_participants = sorted(list(all_participants))
    
    # Create a color map for participants
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(all_participants)))
    participant_colors = {participant: colors[i] for i, participant in enumerate(all_participants)}
    
    # Plot data for each participant
    for participant in all_participants:
        x_points = []
        y_points = []
        
        # Collect data points for this participant
        for i, angle in enumerate(alpha_angles):
            if participant in alpha_thresholds[angle]['participants']:
                idx = alpha_thresholds[angle]['participants'].index(participant)
                x_points.append(angle)
                y_points.append(alpha_thresholds[angle]['delta_threshold'][idx])
        
        # Plot this participant's data
        if len(x_points) > 0:
            plt.scatter(x_points, y_points, color=participant_colors[participant], 
                        label=participant, alpha=0.7)
            
            # Connect points with lines if there are multiple points
            if len(x_points) > 1:
                # Sort points by x value to ensure line connects them properly
                points = sorted(zip(x_points, y_points))
                x_sorted = [p[0] for p in points]
                y_sorted = [p[1] for p in points]
                plt.plot(x_sorted, y_sorted, color=participant_colors[participant], alpha=0.4)
    
    # Add reference line at y=0
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Set labels and title
    plt.xlabel('Alpha Angle (degrees)', fontsize=12)
    plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
    plt.title('Individual Participant Delta Thresholds by Alpha Angle', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.2)
    
    # Handle the legend - if there are many participants, put it outside
    if len(all_participants) > 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust plot area to make room for legend
    else:
        plt.legend(fontsize='small')
        plt.tight_layout()
    
    # Save figure
    output_filename = f"individual_participant_deltas{suffix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Individual participant plot saved to {output_filename}")

# Function to process data and create plots for learners and non-learners
def process_data_by_learning_status(data_dir, file_pattern='*.csv', estimation_method='mean', use_last_five=True):
    """
    Process data and create separate plots for learners and non-learners.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
    file_pattern : str
        Pattern to match CSV files
    estimation_method : str
        Method to estimate threshold (fixed to 'mean')
    use_last_five : bool
        Whether to use only the last 5 trials for threshold estimation (fixed to True)
    """
    # Override parameters to ensure we're using mean and last 5 trials
    estimation_method = 'mean'
    use_last_five = True
    
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
    learner_results = process_group(combined_data, learners, "Learners", estimation_method, use_last_five)
    non_learner_results = process_group(combined_data, non_learners, "Non-Learners", estimation_method, use_last_five)
    
    # Create comparison plots
    if learner_results and non_learner_results:
        create_comparison_plots(learner_results, non_learner_results, estimation_method, use_last_five)
    else:
        print("Skipping comparison plots due to missing data for one or more groups")

def process_group(data, participant_list, group_name, estimation_method='mean', use_last_five=True):
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
        Method to estimate threshold (fixed to 'mean')
    use_last_five : bool
        Whether to use only the last 5 trials for threshold estimation (fixed to True)
        
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
            quest_post_alpha.importData(post_intensities_to_use, post_responses)