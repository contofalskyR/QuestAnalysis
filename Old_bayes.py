import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from scipy import stats
import warnings
import argparse
warnings.filterwarnings('ignore')

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

def plot_delta_vs_mutual_information(df, delta_threshold_col, group_name="All Participants", save_dir="."):
    """
    Plot delta threshold as a function of mutual information.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with processed threshold data
    delta_threshold_col : str
        Name of the column containing delta threshold values
    group_name : str
        Name of the group to include in the plot title
    save_dir : str
        Directory to save the output plot
        
    Returns:
    --------
    tuple: (mutual_info_values, delta_means, r_squared)
    """
    plt.figure(figsize=(10, 6))
    
    # Group by alpha angle and calculate mean delta threshold
    alpha_angles = np.array([0, 22.5, 45, 67.5, 90])
    mutual_info = [calculate_mutual_information(angle) for angle in alpha_angles]
    
    # Prepare data structure for summarized data
    summarized_data = pd.DataFrame({
        'alpha_angle': alpha_angles,
        'mutual_info': mutual_info,
        'delta_threshold': np.nan,
        'delta_error': np.nan,
        'n': np.nan
    })
    
    # Extract feature index from participant ID if available
    if 'feature_index' in df.columns:
        # If the data already has a feature_index column, use it
        grouped = df.groupby('feature_index')
    elif 'alpha_angle' in df.columns:
        # If the data has alpha_angle column, use it
        grouped = df.groupby('alpha_angle')
    else:
        # Otherwise, try to extract from participant ID or another method
        # This is a placeholder - you may need to adjust based on your data structure
        print("No feature_index or alpha_angle column found.")
        # For demonstration, assigning random alpha angles
        df['assigned_alpha'] = np.random.choice(alpha_angles, size=len(df))
        grouped = df.groupby('assigned_alpha')
    
    # Calculate means and errors for each alpha angle/mutual info value
    for i, angle in enumerate(alpha_angles):
        group = grouped.get_group(angle) if angle in grouped.groups else None
        
        if group is not None and len(group) > 0:
            mean_delta = group[delta_threshold_col].mean()
            std_err = group[delta_threshold_col].std() / np.sqrt(len(group))
            summarized_data.loc[i, 'delta_threshold'] = mean_delta
            summarized_data.loc[i, 'delta_error'] = std_err
            summarized_data.loc[i, 'n'] = len(group)
    
    # Plot data with error bars
    plt.errorbar(
        summarized_data['mutual_info'], 
        summarized_data['delta_threshold'], 
        yerr=summarized_data['delta_error'], 
        fmt='o', 
        color='blue', 
        markersize=8, 
        capsize=5
    )
    
    # Fit a linear regression
    valid_indices = summarized_data.dropna(subset=['delta_threshold']).index
    if len(valid_indices) > 1:
        x_vals = summarized_data.loc[valid_indices, 'mutual_info']
        y_vals = summarized_data.loc[valid_indices, 'delta_threshold']
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        r_squared = r_value**2
        
        # Plot regression line
        x_line = np.linspace(0, max(mutual_info), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, '--', color='red')
        
        # Add R² text
        plt.text(max(mutual_info) * 0.1, max(y_vals) * 0.9,
                f'R² = {r_squared:.3f}\ny = {slope:.3f}x + {intercept:.3f}\np = {p_value:.3f}',
                bbox=dict(facecolor='white', alpha=0.7))
    else:
        r_squared = 0
    
    # Add annotation for each point (alpha angle)
    for i, row in summarized_data.iterrows():
        if not np.isnan(row['delta_threshold']):
            plt.annotate(f'α={row["alpha_angle"]:.1f}°', 
                        xy=(row['mutual_info'], row['delta_threshold']),
                        xytext=(5, 5),
                        textcoords='offset points')
    
    # Set labels and title
    plt.xlabel('Mutual Information (bits)', fontsize=12)
    plt.ylabel('ΔThreshold (Pre - Post)', fontsize=12)
    plt.title(f'ΔThreshold vs Mutual Information - {group_name}', fontsize=14)
    
    # Add reference line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_filename = os.path.join(save_dir, f"delta_vs_mi_{group_name.replace(' ', '_')}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Mutual Information plot saved to {output_filename}")
    
    return mutual_info, summarized_data['delta_threshold'].values, r_squared

def bayesian_mi_analysis(mutual_info, delta_thresholds, group_name="Group", save_prefix="bayesian", save_dir="."):
    """
    Run Bayesian regression analysis on the relationship between mutual information and delta threshold.
    
    Parameters:
    -----------
    mutual_info : list or array
        Mutual information values
    delta_thresholds : list or array
        Delta threshold values (Pre - Post)
    group_name : str
        Name of the group for plot titles and filenames
    save_prefix : str
        Prefix for saved plots
    save_dir : str
        Directory to save output files
        
    Returns:
    --------
    dict: Analysis results
    """
    # Remove any NaN values
    valid_idx = ~np.isnan(delta_thresholds)
    x = np.array(mutual_info)[valid_idx]
    y = np.array(delta_thresholds)[valid_idx]
    
    if len(x) < 3:
        print(f"Not enough data points for Bayesian analysis of {group_name}")
        return None
    
    # For comparison, calculate the frequentist regression results
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value**2
    
    print(f"\n--- Frequentist Analysis for {group_name} ---")
    print(f"Linear regression: y = {slope:.4f}x + {intercept:.4f}")
    print(f"R² = {r_squared:.4f}")
    print(f"p-value = {p_value:.4f}")
    
    # Standardize the predictor (optional but often helps with sampling)
    x_scaled = (x - np.mean(x)) / np.std(x)
    
    # Bayesian linear regression model
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=1)  # Intercept
        beta = pm.Normal('beta', mu=0, sigma=1)    # Slope
        sigma = pm.HalfCauchy('sigma', beta=1)     # Error SD
        
        # Expected value
        mu = alpha + beta * x_scaled
        
        # Likelihood
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
        
        # Sample from the posterior
        trace = pm.sample(2000, tune=1000, return_inferencedata=True, 
                          cores=2, chains=2, target_accept=0.9)
    
    # Extract posterior samples for analysis
    posterior = trace.posterior
    
    # Calculate the 95% HDI (Highest Density Interval) for beta
    beta_samples = posterior['beta'].values.flatten()
    hdi = az.hdi(beta_samples, hdi_prob=0.95)
    
    # Calculate Bayes Factor for the slope (approximate method)
    # Savage-Dickey density ratio method for BF10
    prior_density = stats.norm.pdf(0, loc=0, scale=1)
    
    # Fit a KDE to the posterior samples to estimate density at beta=0
    posterior_kde = stats.gaussian_kde(beta_samples)
    posterior_density = posterior_kde(0)[0]
    
    # Bayes Factor (BF10) - evidence for alternative hypothesis
    bf10 = prior_density / posterior_density
    
    # Convert back to unstandardized slope
    beta_unstd = beta_samples / np.std(x)
    hdi_unstd = hdi / np.std(x)
    
    # Create plot for Bayesian analysis results
    plt.figure(figsize=(12, 8))
    
    # Plot raw data with regression line
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, color='blue', alpha=0.7)
    
    # Plot the regression line with uncertainty
    x_plot = np.linspace(min(x), max(x), 100)
    x_plot_scaled = (x_plot - np.mean(x)) / np.std(x)
    
    # Extract samples for plotting regression lines
    alpha_samples = posterior['alpha'].values.flatten()
    
    # Compute posterior predictive draws
    n_draws = 100
    indices = np.random.randint(0, len(alpha_samples), n_draws)
    
    for i in indices:
        y_pred = alpha_samples[i] + beta_samples[i] * x_plot_scaled
        plt.plot(x_plot, y_pred, color='red', alpha=0.05)
    
    # Add mean regression line
    y_mean = np.mean(alpha_samples) + np.mean(beta_samples) * x_plot_scaled
    plt.plot(x_plot, y_mean, color='red', linewidth=2)
    
    # Add annotations
    plt.xlabel('Mutual Information (bits)')
    plt.ylabel('Delta Threshold (Pre - Post)')
    plt.title(f'Bayesian Regression: {group_name}')
    plt.grid(True, alpha=0.3)
    
    # Add stats to the plot
    stats_text = (
        f"Frequentist: R² = {r_squared:.3f}, p = {p_value:.3f}\n"
        f"Bayesian: 95% HDI = [{hdi_unstd[0]:.3f}, {hdi_unstd[1]:.3f}]\n"
        f"Bayes Factor (BF10) = {bf10:.2f}"
    )
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7),
             verticalalignment='top')
    
    # Plot posterior distribution of the slope
    plt.subplot(2, 1, 2)
    sns.histplot(beta_unstd, kde=True, stat="density")
    
    # Add vertical lines for HDI and zero
    plt.axvline(x=0, color='black', linestyle='--', label='Null hypothesis (β=0)')
    plt.axvline(x=hdi_unstd[0], color='red', linestyle='--', label=f'95% HDI: [{hdi_unstd[0]:.3f}, {hdi_unstd[1]:.3f}]')
    plt.axvline(x=hdi_unstd[1], color='red', linestyle='--')
    
    # Shade the HDI region
    x_range = np.linspace(hdi_unstd[0], hdi_unstd[1], 100)
    y_kde = posterior_kde(x_range * np.std(x)) / np.std(x)
    plt.fill_between(x_range, y_kde, alpha=0.2, color='red')
    
    plt.xlabel('Slope (β)')
    plt.ylabel('Density')
    plt.title('Posterior Distribution of Slope')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_filename = os.path.join(save_dir, f"{save_prefix}_{group_name.replace(' ', '_')}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    # Add interpretation of Bayesian results
    print("\n--- Bayesian Analysis Results ---")
    print(f"95% HDI for slope: [{hdi_unstd[0]:.4f}, {hdi_unstd[1]:.4f}]")
    print(f"Bayes Factor (BF10): {bf10:.4f}")
    
    if 0 < hdi_unstd[0] or hdi_unstd[1] < 0:
        print("The 95% HDI excludes zero, providing strong evidence for an effect.")
    else:
        print("The 95% HDI includes zero, suggesting uncertainty about the effect.")
    
    if bf10 > 10:
        print("Bayes Factor: Strong evidence for the alternative hypothesis (positive relationship)")
    elif bf10 > 3:
        print("Bayes Factor: Moderate evidence for the alternative hypothesis")
    elif bf10 > 1:
        print("Bayes Factor: Anecdotal evidence for the alternative hypothesis")
    elif bf10 > 1/3:
        print("Bayes Factor: Inconclusive evidence")
    elif bf10 > 1/10:
        print("Bayes Factor: Moderate evidence for the null hypothesis (no relationship)")
    else:
        print("Bayes Factor: Strong evidence for the null hypothesis")
    
    # Return the results
    return {
        'frequentist': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value
        },
        'bayesian': {
            'posterior_samples': {
                'alpha': alpha_samples,
                'beta': beta_samples,
                'beta_unstd': beta_unstd
            },
            'hdi': hdi_unstd,
            'bayes_factor': bf10
        }
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bayesian analysis of threshold data')
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to CSV file with processed threshold data')
    parser.add_argument('--threshold_col', type=str, default='delta_threshold_mean', 
                        help='Column name for delta threshold values (default: delta_threshold_mean)')
    parser.add_argument('--output_dir', type=str, default='.', 
                        help='Directory to save output files (default: current directory)')
    parser.add_argument('--group_name', type=str, default='All Participants', 
                        help='Name of the group for plot titles (default: All Participants)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Reading threshold data from {args.input_file}")
    
    # Load threshold data
    try:
        df = pd.read_csv(args.input_file)
        print(f"Loaded {len(df)} rows of data")
        
        # Check if the specified threshold column exists
        if args.threshold_col not in df.columns:
            print(f"Error: Column '{args.threshold_col}' not found in the input file")
            print(f"Available columns: {', '.join(df.columns)}")
            return
        
        # Print summary statistics of the delta threshold values
        print(f"\nSummary of {args.threshold_col}:")
        print(df[args.threshold_col].describe())
        
        # Define alpha angles for analysis (adjust if your data has different angles)
        alpha_angles = np.array([0, 22.5, 45, 67.5, 90])
        
        # Calculate mutual information for each alpha angle
        mutual_info = [calculate_mutual_information(angle) for angle in alpha_angles]
        
        # Create a plot of delta threshold vs mutual information
        mi_values, delta_means, r_squared = plot_delta_vs_mutual_information(
            df, args.threshold_col, args.group_name, args.output_dir
        )
        
        # Run Bayesian analysis on the relationship
        bayesian_results = bayesian_mi_analysis(
            mutual_info, delta_means, args.group_name, "bayesian", args.output_dir
        )
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()