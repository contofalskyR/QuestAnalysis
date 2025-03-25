import pandas as pd
import os

# Path to your Bayesian CSV file
bayesian_csv_path = '/Users/robertrutgers/Documents/2Category-Alpha Data/NewAlpha/quest_thresholds_NewAlpha_by_feature.csv'

# Create output directory if it doesn't exist
output_dir = '/Users/robertrutgers/Documents/2Category-Alpha Data/NewAlpha/bayesian_analysis'
os.makedirs(output_dir, exist_ok=True)

# List of learner participant IDs (from your log file)
# Add all your learner IDs here
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

try:
    # Read the Bayesian CSV
    bayesian_df = pd.read_csv(bayesian_csv_path)
    print(f"Loaded Bayesian CSV with {len(bayesian_df)} rows")
    
    # Print columns to help diagnose the structure
    print(f"Columns in Bayesian CSV: {bayesian_df.columns.tolist()}")
    print(f"First row: {bayesian_df.iloc[0]}")
    
    # Add learner status column
    bayesian_df['is_learner'] = False
    
    # Check if there's a participant column or if we need to use another column
    if 'participant' in bayesian_df.columns:
        # Update learner status based on participant ID
        for i, row in bayesian_df.iterrows():
            if row['participant'] in learner_ids:
                bayesian_df.at[i, 'is_learner'] = True
    else:
        print("No 'participant' column found. Please specify which column contains participant IDs:")
        participant_column = input("Enter column name: ")
        
        # Update learner status based on specified column
        for i, row in bayesian_df.iterrows():
            # Check if the participant ID is in (or contains) a learner ID
            for learner_id in learner_ids:
                if learner_id in str(row[participant_column]):
                    bayesian_df.at[i, 'is_learner'] = True
                    break
    
    # Split into learners and non-learners
    learners_df = bayesian_df[bayesian_df['is_learner'] == True]
    non_learners_df = bayesian_df[bayesian_df['is_learner'] == False]
    
    # Save to separate CSV files
    all_output_path = os.path.join(output_dir, 'all_participants.csv')
    learners_output_path = os.path.join(output_dir, 'learners.csv')
    non_learners_output_path = os.path.join(output_dir, 'non_learners.csv')
    
    bayesian_df.to_csv(all_output_path, index=False)
    learners_df.to_csv(learners_output_path, index=False)
    non_learners_df.to_csv(non_learners_output_path, index=False)
    
    print(f"Saved all participants ({len(bayesian_df)} rows) to: {all_output_path}")
    print(f"Saved {len(learners_df)} learner rows to: {learners_output_path}")
    print(f"Saved {len(non_learners_df)} non-learner rows to: {non_learners_output_path}")
    
except Exception as e:
    print(f"Error processing Bayesian CSV: {str(e)}")
    import traceback
    traceback.print_exc()