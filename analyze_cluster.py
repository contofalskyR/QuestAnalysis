#!/usr/bin/env python
"""
Script to analyze a specific cluster in detail
"""
import pandas as pd
import os
from expert_analysis import analyze_cluster_details, perceptual_strategy_analysis

# Load the data
df = pd.read_csv('quest_thresholds_NewAlpha_by_feature.csv')

# First run the strategy analysis to get the expert profiles
strategy_results = perceptual_strategy_analysis(
    df,
    participant_col='participant',
    feature_col='feature_index',
    pre_col='pre_threshold_mean',
    post_col='post_threshold_mean',
    delta_col='delta_threshold_mean',
    save_dir='./results'
)

# Then analyze Cluster 1 in detail
if strategy_results and 'expert_profiles' in strategy_results:
    cluster_details = analyze_cluster_details(
        df, 
        strategy_results['expert_profiles'],
        cluster_id=1,
        participant_col='participant',
        feature_col='feature_index',
        pre_col='pre_threshold_mean',
        post_col='post_threshold_mean',
        save_dir='./results'
    )
    print("Detailed analysis of Cluster 1 complete.")
else:
    print("Strategy analysis did not return expert profiles.")