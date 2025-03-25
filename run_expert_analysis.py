#!/usr/bin/env python
"""
Expert Analysis for Categorical Perception
-----------------------------------------
This script runs comprehensive analysis of expert vs. non-expert patterns
in categorical perception data, exploring the "expert paradox" phenomenon.

Usage:
    python run_expert_analysis.py --input all_participants.csv --output ./results
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from expert_analysis import (
    pattern_analysis,
    perceptual_strategy_analysis,
    learning_dynamics_analysis,
    alternative_information_analysis,
    analyze_expert_data
)

def setup_environment():
    """Set up the environment for the analysis."""
    # Configure matplotlib for better visualizations
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configure pandas display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

def validate_data(df, required_cols):
    """Validate the input data has necessary columns."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    
    # Check for expert/learner participants
    if 'participant' in df.columns:
        from expert_analysis import learner_ids
        n_experts = len([p for p in df['participant'].unique() if p in learner_ids])
        if n_experts == 0:
            print("WARNING: No expert/learner participants found in the data.")
            print("Please check learner_ids in expert_analysis.py")
            return False
    
    return True

def main():
    """Main function to run the analysis."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run expert analysis on categorical perception data')
    parser.add_argument('--input', '-i', required=True, help='Path to CSV file with threshold data')
    parser.add_argument('--output', '-o', default='./results', help='Directory to save results (default: ./results)')
    parser.add_argument('--analysis', '-a', choices=['all', 'pattern', 'strategy', 'dynamics', 'metrics'],
                       default='all', help='Which analysis to run (default: all)')
    
    args = parser.parse_args()
    
    # Setup
    setup_environment()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file '{args.input}' not found")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"ERROR: Failed to load data: {str(e)}")
        return 1
    
    # Validate data
    required_cols = ['participant', 'feature_index', 
                    'pre_threshold_mean', 'post_threshold_mean', 'delta_threshold_mean']
    if not validate_data(df, required_cols):
        return 1
    
    # Run the selected analysis
    if args.analysis == 'all':
        print("\nRunning ALL analyses...")
        results = analyze_expert_data(args.input, save_dir=args.output)
    else:
        # Run specific analysis
        if args.analysis == 'pattern':
            print("\nRunning PATTERN analysis...")
            results = pattern_analysis(
                df,
                participant_col='participant',
                feature_col='feature_index',
                pre_col='pre_threshold_mean',
                post_col='post_threshold_mean',
                delta_col='delta_threshold_mean',
                save_dir=args.output
            )
        elif args.analysis == 'strategy':
            print("\nRunning STRATEGY analysis...")
            results = perceptual_strategy_analysis(
                df,
                participant_col='participant',
                feature_col='feature_index',
                pre_col='pre_threshold_mean',
                post_col='post_threshold_mean',
                delta_col='delta_threshold_mean',
                save_dir=args.output
            )
            if strategy_results and 'expert_profiles' in strategy_results:
                # Analyze cluster 1 (the small cluster with unusual pattern)
                cluster_details = analyze_cluster_details(
                    df,
                    strategy_results['expert_profiles'],
                    cluster_id=1,  # Cluster with unusual negative delta thresholds
                    participant_col='participant',
                    feature_col='feature_index',
                    pre_col='pre_threshold_mean',
                    post_col='post_threshold_mean',
                    save_dir=args.output
                )
        elif args.analysis == 'dynamics':
            print("\nRunning DYNAMICS analysis...")
            results = learning_dynamics_analysis(
                df,
                participant_col='participant',
                feature_col='feature_index',
                pre_col='pre_threshold_mean',
                post_col='post_threshold_mean',
                save_dir=args.output
            )
        elif args.analysis == 'metrics':
            print("\nRunning METRICS analysis...")
            results = alternative_information_analysis(
                df,
                participant_col='participant',
                feature_col='feature_index',
                pre_col='pre_threshold_mean',
                post_col='post_threshold_mean',
                delta_col='delta_threshold_mean',
                save_dir=args.output
            )
    
    print(f"\nAnalysis complete. Results saved to {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())