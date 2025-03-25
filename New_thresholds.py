import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import os

# Load the data from the provided text
data_text = """participant	feature_index	alpha_angle	learner_status	pre_threshold	post_threshold	delta_threshold	pre_threshold_last5	post_threshold_last5	delta_threshold_last5
848799_Utility_discrimination_2024_2cat_2025-03-12_13h57.25.042	0	0	Learner	0.343282858	0.248830828	0.09445203	-0.034636243	-0.045767401	0.011131158
848799_Utility_discrimination_2024_2cat_2025-03-12_13h57.25.042	1	22.5	Learner	0.343455754	0.190523328	0.152932426	0.213762491	0.155242658	0.058519833
848799_Utility_discrimination_2024_2cat_2025-03-12_13h57.25.042	2	45	Learner	0.470792206	0.320924913	0.149867293	0.321717615	0.181048301	0.140669314
848799_Utility_discrimination_2024_2cat_2025-03-12_13h57.25.042	3	67.5	Learner	0.342386762	0.171442356	0.170944406	0.195631999	0.179865115	0.015766884
848799_Utility_discrimination_2024_2cat_2025-03-12_13h57.25.042	4	90	Learner	0.486461594	0.323751745	0.162709849	0.208373041	0.311375735	-0.103002694
509243_Utility_discrimination_2024_2cat_2025-03-14_00h08.13.599	0	0	Non-Learner	1.087174095	1.206061052	-0.118886956	0.376944311	0.626892263	-0.249947952
509243_Utility_discrimination_2024_2cat_2025-03-14_00h08.13.599	1	22.5	Non-Learner	0.939615563	1.128503417	-0.188887853	0.334798132	0.458601189	-0.123803057
509243_Utility_discrimination_2024_2cat_2025-03-14_00h08.13.599	2	45	Non-Learner	1.092584549	1.234338613	-0.141754064	0.610187575	0.314863606	0.295323969
509243_Utility_discrimination_2024_2cat_2025-03-14_00h08.13.599	3	67.5	Non-Learner	1.040796229	1.317749986	-0.276953756	0.685410196	0.382597527	0.302812669
509243_Utility_discrimination_2024_2cat_2025-03-14_00h08.13.599	4	90	Non-Learner	1.036802435	1.353978774	-0.317176339	0.903935015	0.157168357	0.746766658
225967_Utility_discrimination_2024_2cat_2025-03-14_00h37.01.876	0	0	Non-Learner	1.744470091	1.428718997	0.315751094	1.164434332	0.665624485	0.498809847
225967_Utility_discrimination_2024_2cat_2025-03-14_00h37.01.876	1	22.5	Non-Learner	1.650491692	1.592652092	0.0578396	0.233637583	1.21861459	-0.984977007
225967_Utility_discrimination_2024_2cat_2025-03-14_00h37.01.876	2	45	Non-Learner	1.839423382	1.708603669	0.130819713	1.255729866	0.137019445	1.118710421
225967_Utility_discrimination_2024_2cat_2025-03-14_00h37.01.876	3	67.5	Non-Learner	1.791356331	1.584006105	0.207350225	1.502813042	0.139890006	1.362923037
225967_Utility_discrimination_2024_2cat_2025-03-14_00h37.01.876	4	90	Non-Learner	1.636160691	1.739519456	-0.103358765	0.2902796	0.912061386	-0.621781786
516600_Utility_discrimination_2024_2cat_2025-03-12_14h14.48.442	0	0	Non-Learner	0.163629612	1.329758278	-1.166128666	0.445076193	0.027131705	0.417944488
516600_Utility_discrimination_2024_2cat_2025-03-12_14h14.48.442	1	22.5	Non-Learner	0.093032027	0.63206885	-0.539036823	0.346855116	0.215152361	0.131702756
516600_Utility_discrimination_2024_2cat_2025-03-12_14h14.48.442	2	45	Non-Learner	0.076540402	1.476909223	-1.400368821	0.076253941	0.014009123	0.062244819
516600_Utility_discrimination_2024_2cat_2025-03-12_14h14.48.442	3	67.5	Non-Learner	0.315863122	0.970278176	-0.654415054	0.300160544	0.037679583	0.262480962
516600_Utility_discrimination_2024_2cat_2025-03-12_14h14.48.442	4	90	Non-Learner	0.324608722	1.243675009	-0.919066287	0.24384442	0.535635529	-0.291791109
227467_Utility_discrimination_2024_2cat_2025-03-13_18h34.20.777	0	0	Learner	-0.845594232	-1.131985857	0.286391625	-0.025150296	0.024912449	-0.050062745
227467_Utility_discrimination_2024_2cat_2025-03-13_18h34.20.777	1	22.5	Learner	-0.427654026	-1.246012348	0.818358322	-0.025178842	0.234679775	-0.259858618
227467_Utility_discrimination_2024_2cat_2025-03-13_18h34.20.777	2	45	Learner	-1.058031038	-1.399893539	0.341862502	0.159603533	0.016487974	0.14311556
227467_Utility_discrimination_2024_2cat_2025-03-13_18h34.20.777	3	67.5	Learner	-0.415167139	0.029140089	-0.444307229	0.033860727	0.233725088	-0.199864361
227467_Utility_discrimination_2024_2cat_2025-03-13_18h34.20.777	4	90	Learner	-0.581587881	-1.123606414	0.542018533	-0.077080753	0.021907104	-0.098987857
838091_Utility_discrimination_2024_2cat_2025-03-13_21h26.42.203	0	0	Learner	0.307189656	0.332694388	-0.025504732	-0.044013568	-0.040881915	-0.003131654
838091_Utility_discrimination_2024_2cat_2025-03-13_21h26.42.203	1	22.5	Learner	0.458136201	0.247854871	0.210281329	0.02306509	-0.043406971	0.06647206
838091_Utility_discrimination_2024_2cat_2025-03-13_21h26.42.203	2	45	Learner	0.301790566	0.291288979	0.010501587	-0.043550585	0.482641964	-0.526192548
838091_Utility_discrimination_2024_2cat_2025-03-13_21h26.42.203	3	67.5	Learner	0.317969373	0.273375887	0.044593486	0.204438302	0.16945449	0.034983812
838091_Utility_discrimination_2024_2cat_2025-03-13_21h26.42.203	4	90	Learner	0.189269197	0.43990058	-0.250631383	0.173221732	-0.046763556	0.219985289"""

# Create DataFrame from the data
import io
df = pd.read_csv(io.StringIO(data_text), sep='\t')

# Create an output directory for saved plots
output_dir = "analysis_plots"
os.makedirs(output_dir, exist_ok=True)

# Extract participant IDs for better labeling
df['participant_id'] = df['participant'].str.split('_').str[0]

# Calculate mutual information for each alpha angle
def calculate_mutual_information(alpha_angle, sigma=0.15):
    """Calculate mutual information between category and feature at angle alpha"""
    alpha_rad = math.radians(alpha_angle)
    max_mi = 0.71  # Approx. value for 95% IPL from Feldman's paper
    mi = max_mi * abs(math.cos(alpha_rad))
    return max(0, mi)

# Add mutual information to the dataframe
df['mutual_information'] = df['alpha_angle'].apply(calculate_mutual_information)

# Basic statistics
print("Dataset Summary:")
print(f"Total observations: {len(df)}")
print(f"Number of participants: {df['participant_id'].nunique()}")
print(f"Learners: {df[df['learner_status'] == 'Learner']['participant_id'].nunique()}")
print(f"Non-Learners: {df[df['learner_status'] == 'Non-Learner']['participant_id'].nunique()}")

# Calculate thresholds by alpha angle for all participants
alpha_summary_all = df.groupby('alpha_angle').agg({
    'pre_threshold': ['mean', 'std', 'count'],
    'post_threshold': ['mean', 'std'],
    'delta_threshold': ['mean', 'std'],
    'mutual_information': 'first'
}).reset_index()

# Flatten multi-index columns
alpha_summary_all.columns = ['_'.join(col).strip('_') for col in alpha_summary_all.columns.values]

# Separate by learner status
learners_df = df[df['learner_status'] == 'Learner']
nonlearners_df = df[df['learner_status'] == 'Non-Learner']

# Calculate summaries for learners and non-learners
alpha_summary_learners = learners_df.groupby('alpha_angle').agg({
    'pre_threshold': ['mean', 'std', 'count'],
    'post_threshold': ['mean', 'std'],
    'delta_threshold': ['mean', 'std'],
    'mutual_information': 'first'
}).reset_index()
alpha_summary_learners.columns = ['_'.join(col).strip('_') for col in alpha_summary_learners.columns.values]

alpha_summary_nonlearners = nonlearners_df.groupby('alpha_angle').agg({
    'pre_threshold': ['mean', 'std', 'count'],
    'post_threshold': ['mean', 'std'],
    'delta_threshold': ['mean', 'std'],
    'mutual_information': 'first'
}).reset_index()
alpha_summary_nonlearners.columns = ['_'.join(col).strip('_') for col in alpha_summary_nonlearners.columns.values]

# Calculate correlations with mutual information
corr_all_mi = stats.pearsonr(df['mutual_information'], df['delta_threshold'])
corr_learners_mi = stats.pearsonr(learners_df['mutual_information'], learners_df['delta_threshold'])
corr_nonlearners_mi = stats.pearsonr(nonlearners_df['mutual_information'], nonlearners_df['delta_threshold'])

# Calculate correlations with alpha angle
corr_all_alpha = stats.pearsonr(df['alpha_angle'], df['delta_threshold'])
corr_learners_alpha = stats.pearsonr(learners_df['alpha_angle'], learners_df['delta_threshold'])
corr_nonlearners_alpha = stats.pearsonr(nonlearners_df['alpha_angle'], nonlearners_df['delta_threshold'])

print("\nCorrelations with Mutual Information:")
print(f"All participants: r={corr_all_mi[0]:.3f}, p={corr_all_mi[1]:.3f}")
print(f"Learners: r={corr_learners_mi[0]:.3f}, p={corr_learners_mi[1]:.3f}")
print(f"Non-learners: r={corr_nonlearners_mi[0]:.3f}, p={corr_nonlearners_mi[1]:.3f}")

print("\nCorrelations with Alpha Angle:")
print(f"All participants: r={corr_all_alpha[0]:.3f}, p={corr_all_alpha[1]:.3f}")
print(f"Learners: r={corr_learners_alpha[0]:.3f}, p={corr_learners_alpha[1]:.3f}")
print(f"Non-learners: r={corr_nonlearners_alpha[0]:.3f}, p={corr_nonlearners_alpha[1]:.3f}")

# Print summary table
print("\nSummary by Alpha Angle (Mean Delta Thresholds):")
summary_table = pd.DataFrame({
    'Alpha': alpha_summary_all['alpha_angle'],
    'MI': alpha_summary_all['mutual_information_first'],
    'All_Delta': alpha_summary_all['delta_threshold_mean'].round(3),
    'All_n': alpha_summary_all['pre_threshold_count'],
    'Learners_Delta': alpha_summary_learners['delta_threshold_mean'].round(3),
    'Learners_n': alpha_summary_learners['pre_threshold_count'],
    'NonLearners_Delta': alpha_summary_nonlearners['delta_threshold_mean'].round(3),
    'NonLearners_n': alpha_summary_nonlearners['pre_threshold_count']
})
print(summary_table)

# PLOT 1: Delta threshold as a function of alpha - All Participants
plt.figure(figsize=(10, 6))

# Plot with error bars
plt.errorbar(
    alpha_summary_all['alpha_angle'], 
    alpha_summary_all['delta_threshold_mean'],
    yerr=alpha_summary_all['delta_threshold_std'] / np.sqrt(alpha_summary_all['pre_threshold_count']),
    fmt='o-', color='blue', linewidth=2, markersize=8, capsize=5
)

# Add reference line at y=0
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Add correlation info
plt.figtext(
    0.2, 0.85,
    f"Correlation with Alpha: {corr_all_alpha[0]:.3f}",
    bbox=dict(facecolor='white', alpha=0.7)
)

# For each point, add the number of participants
for i, row in alpha_summary_all.iterrows():
    if not np.isnan(row['delta_threshold_mean']):
        plt.text(row['alpha_angle'], 
               row['delta_threshold_mean'] + (0.01 if row['delta_threshold_mean'] >= 0 else -0.03), 
               f"n={int(row['pre_threshold_count'])}", ha='center')

# Labels and formatting
plt.xlabel('Feature Index (Alpha Angle)', fontsize=12)
plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
plt.title('Delta Threshold as a Function of Alpha - All Participants', fontsize=14)
plt.xticks(alpha_summary_all['alpha_angle'], [f"Feature{i}\n(α={a}°)" for i, a in enumerate(alpha_summary_all['alpha_angle'])])
plt.grid(True, alpha=0.3)

# Add annotation
plt.figtext(
    0.5, 0.01,
    'Positive delta values indicate improvement after discrimination training.\n'
    'Delta threshold is expected to diminish with increasing alpha (angle from max-info dimension).',
    ha='center',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(os.path.join(output_dir, 'delta_threshold_vs_alpha_all.png'), dpi=300)
print(f"Saved plot to {os.path.join(output_dir, 'delta_threshold_vs_alpha_all.png')}")

# PLOT 2: Learners vs Non-Learners comparison
plt.figure(figsize=(10, 6))

# Learners
plt.errorbar(
    alpha_summary_learners['alpha_angle'] - 0.5, 
    alpha_summary_learners['delta_threshold_mean'],
    yerr=alpha_summary_learners['delta_threshold_std'] / np.sqrt(alpha_summary_learners['pre_threshold_count']),
    fmt='o-', color='blue', linewidth=2, markersize=8, capsize=5, label='Learners'
)

# Non-learners
plt.errorbar(
    alpha_summary_nonlearners['alpha_angle'] + 0.5, 
    alpha_summary_nonlearners['delta_threshold_mean'],
    yerr=alpha_summary_nonlearners['delta_threshold_std'] / np.sqrt(alpha_summary_nonlearners['pre_threshold_count']),
    fmt='s-', color='red', linewidth=2, markersize=8, capsize=5, label='Non-Learners'
)

# Add reference line at y=0
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Add correlation info
plt.figtext(
    0.2, 0.85,
    f"Correlations with Alpha:\nLearners: {corr_learners_alpha[0]:.3f}\nNon-Learners: {corr_nonlearners_alpha[0]:.3f}",
    bbox=dict(facecolor='white', alpha=0.7)
)

# Labels and formatting
plt.xlabel('Feature Index (Alpha Angle)', fontsize=12)
plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
plt.title('Delta Threshold as a Function of Alpha - Learners vs Non-Learners', fontsize=14)
plt.xticks(alpha_summary_all['alpha_angle'], [f"Feature{i}\n(α={a}°)" for i, a in enumerate(alpha_summary_all['alpha_angle'])])
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'delta_threshold_vs_alpha_comparison.png'), dpi=300)
print(f"Saved plot to {os.path.join(output_dir, 'delta_threshold_vs_alpha_comparison.png')}")

# PLOT 3: Delta threshold vs Mutual Information
plt.figure(figsize=(10, 6))

# Plot all participants
plt.scatter(
    df['mutual_information'], 
    df['delta_threshold'],
    color='purple', alpha=0.5, label='All Participants'
)

# Add trendline
x_mi = np.linspace(0, max(df['mutual_information']), 100)
z = np.polyfit(df['mutual_information'], df['delta_threshold'], 1)
p = np.poly1d(z)
plt.plot(x_mi, p(x_mi), '--', color='purple')

# Separate by learner status
plt.scatter(
    learners_df['mutual_information'], 
    learners_df['delta_threshold'],
    color='blue', marker='o', label='Learners'
)
 
plt.scatter(
    nonlearners_df['mutual_information'], 
    nonlearners_df['delta_threshold'],
    color='red', marker='s', label='Non-Learners'
)

# Add trendlines for learners and non-learners
z_l = np.polyfit(learners_df['mutual_information'], learners_df['delta_threshold'], 1)
p_l = np.poly1d(z_l)
plt.plot(x_mi, p_l(x_mi), '--', color='blue')

z_nl = np.polyfit(nonlearners_df['mutual_information'], nonlearners_df['delta_threshold'], 1)
p_nl = np.poly1d(z_nl)
plt.plot(x_mi, p_nl(x_mi), '--', color='red')

# Reference line at y=0
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Formatting
plt.xlabel('Mutual Information (bits)', fontsize=12)
plt.ylabel('Delta Threshold (Pre - Post)', fontsize=12)
plt.title('Delta Threshold vs Mutual Information', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Add correlation values
y_pos = 0.85
plt.figtext(
    0.2, y_pos,
    f"All: r={corr_all_mi[0]:.3f}, R²={corr_all_mi[0]**2:.3f}",
    color='purple',
    bbox=dict(facecolor='white', alpha=0.7)
)

y_pos -= 0.05
plt.figtext(
    0.2, y_pos,
    f"Learners: r={corr_learners_mi[0]:.3f}, R²={corr_learners_mi[0]**2:.3f}",
    color='blue',
    bbox=dict(facecolor='white', alpha=0.7)
)

y_pos -= 0.05
plt.figtext(
    0.2, y_pos,
    f"Non-Learners: r={corr_nonlearners_mi[0]:.3f}, R²={corr_nonlearners_mi[0]**2:.3f}",
    color='red',
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'delta_threshold_vs_mutual_information.png'), dpi=300)
print(f"Saved plot to {os.path.join(output_dir, 'delta_threshold_vs_mutual_information.png')}")

# PLOT 4: Pre and Post thresholds comparison
plt.figure(figsize=(12, 6))

# Learners
plt.plot(alpha_summary_learners['alpha_angle'], alpha_summary_learners['pre_threshold_mean'], 
         'o-', color='darkblue', label='Learners - Pre')
plt.plot(alpha_summary_learners['alpha_angle'], alpha_summary_learners['post_threshold_mean'], 
         's-', color='blue', label='Learners - Post')

# Non-learners
plt.plot(alpha_summary_nonlearners['alpha_angle'], alpha_summary_nonlearners['pre_threshold_mean'], 
         'o-', color='darkred', label='Non-Learners - Pre')
plt.plot(alpha_summary_nonlearners['alpha_angle'], alpha_summary_nonlearners['post_threshold_mean'], 
         's-', color='red', label='Non-Learners - Post')

# Formatting
plt.xlabel('Feature Index (Alpha Angle)', fontsize=12)
plt.ylabel('Threshold', fontsize=12)
plt.title('Pre and Post Discrimination Thresholds by Alpha - Learners vs Non-Learners', fontsize=14)
plt.xticks(alpha_summary_all['alpha_angle'], [f"Feature{i}\n(α={a}°)" for i, a in enumerate(alpha_summary_all['alpha_angle'])])
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pre_post_thresholds_comparison.png'), dpi=300)
print(f"Saved plot to {os.path.join(output_dir, 'pre_post_thresholds_comparison.png')}")

# PLOT 5: Individual participant data by learner status
plt.figure(figsize=(14, 8))

# Create subplots for learners and non-learners
plt.subplot(1, 2, 1)
learner_ids = learners_df['participant_id'].unique()
for idx, participant_id in enumerate(learner_ids):
    participant_data = learners_df[learners_df['participant_id'] == participant_id]
    plt.plot(
        participant_data['alpha_angle'],
        participant_data['delta_threshold'],
        'o-',
        label=participant_id
    )
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.xlabel('Alpha Angle (degrees)')
plt.ylabel('Delta Threshold (Pre - Post)')
plt.title('Learners: Individual Participant Data')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
nonlearner_ids = nonlearners_df['participant_id'].unique()
for idx, participant_id in enumerate(nonlearner_ids):
    participant_data = nonlearners_df[nonlearners_df['participant_id'] == participant_id]
    plt.plot(
        participant_data['alpha_angle'],
        participant_data['delta_threshold'],
        's-',
        label=participant_id
    )
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.xlabel('Alpha Angle (degrees)')
plt.ylabel('Delta Threshold (Pre - Post)')
plt.title('Non-Learners: Individual Participant Data')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'individual_participants.png'), dpi=300)
print(f"Saved plot to {os.path.join(output_dir, 'individual_participants.png')}")

# Create a findings summary
print("\n=== FINDINGS SUMMARY ===")
print("\nComparison to Feldman's findings:")
print(f"1. Feldman found strong correlation between delta threshold and mutual information (R² = .5663)")
print(f"2. Your data shows:")
print(f"   All participants: r={corr_all_mi[0]:.3f}, R²={corr_all_mi[0]**2:.3f}")
print(f"   Learners: r={corr_learners_mi[0]:.3f}, R²={corr_learners_mi[0]**2:.3f}")
print(f"   Non-learners: r={corr_nonlearners_mi[0]:.3f}, R²={corr_nonlearners_mi[0]**2:.3f}")

# Check if the correlations match Feldman's direction
feldman_direction = "consistent" if corr_all_mi[0] > 0 else "opposite"
print(f"\n3. Direction of correlation is {feldman_direction} with Feldman's findings")

print("\n4. Delta threshold patterns by alpha angle:")
for angle in alpha_summary_all['alpha_angle']:
    angle_data = alpha_summary_all[alpha_summary_all['alpha_angle'] == angle]
    learner_data = alpha_summary_learners[alpha_summary_learners['alpha_angle'] == angle]
    nonlearner_data = alpha_summary_nonlearners[alpha_summary_nonlearners['alpha_angle'] == angle]
    
    print(f"   α={angle}°: All={angle_data['delta_threshold_mean'].values[0]:.3f}, " +
          f"Learners={learner_data['delta_threshold_mean'].values[0]:.3f}, " +
          f"Non-Learners={nonlearner_data['delta_threshold_mean'].values[0]:.3f}")

print("\n5. Key observations:")
print("   - All plots have been saved as PNG files in the 'analysis_plots' directory")
print("   - Individual participant data shows high variability")
print(f"   - Learners tend to show {'positive' if learners_df['delta_threshold'].mean() > 0 else 'negative'} delta thresholds overall")
print(f"   - Non-learners tend to show {'positive' if nonlearners_df['delta_threshold'].mean() > 0 else 'negative'} delta thresholds overall")
print(f"   - Sample size (n={df['participant_id'].nunique()}) is smaller than Feldman's experiments (n=20+)")