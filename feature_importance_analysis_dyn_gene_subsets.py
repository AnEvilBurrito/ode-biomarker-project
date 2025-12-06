# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: ode-biomarker-project
#     language: python
#     name: python3
# ---

# %%
# Jupyter notebook in Jupytext format

# %% [markdown]
# ## Feature Importance Analysis for Network-Specific Gene Subsets
# ## Cross-Dataset Visualizations with Publication-Quality Plots

# %% [markdown]
# This notebook performs comprehensive analysis of feature importance results from the gene subset SHAP analysis and generates publication-quality visualizations including:
# - Matrix plots comparing feature importance across networks and dataset types (gene subset vs dynamic vs combined)
# - Paired comparison plots showing context-dependent importance changes
# - SHAP directional impact analysis with natural language titles emphasizing network-specific gene subsets
# - Cross-dataset pattern identification for targeted biomarker discovery

# %% [markdown]
# ## Initialisation and Configuration

# %%
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to Python path for imports
path = os.getcwd()
# find the string 'project' in the path, return index
index_project = path.find("project")
# slice the path from the index of 'project' to the end
project_path = path[: index_project + 7]
# set the working directory
os.chdir(project_path)
print(f"Project path set to: {os.getcwd()}")

# Add project root to Python path for imports
sys.path.insert(0, project_path)

# %%
from PathLoader import PathLoader #noqa: E402

path_loader = PathLoader("data_config.env", "current_user.env")

# %%
from DataLink import DataLink #noqa: E402

data_link = DataLink(path_loader, "data_codes.csv")

# %%
# Use the gene subset folder structure instead of original
folder_name = "ThesisResult-FeatureImportanceGeneSubsets-SHAP"
exp_id = "v1_rf_config1_genesubsets_shap_seeds20_batch4" 

# Create both the main folder and exp_id subfolder
main_folder = f"{path_loader.get_data_path()}data/results/{folder_name}"
exp_folder = f"{main_folder}/{exp_id}"

if not os.path.exists(main_folder):
    os.makedirs(main_folder)
if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

file_save_path = f"{exp_folder}/"

# Create a new report file for capturing print statements
print_report_path = f"{file_save_path}feature_importance_gene_subset_analysis_report_{exp_id}.md"
print_report_file = open(print_report_path, 'w', encoding='utf-8')

# Write header to the print report
print_report_file.write(f"# Feature Importance Analysis Report - Gene Subsets - {exp_id}\n\n")
import time
print_report_file.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
print_report_file.write("This report captures all print statements from the feature importance analysis for network-specific gene subsets with proper formatting.\n\n")

def save_and_print(message, report_file=None, level="info"):
    """
    Print message to console and save to report file with proper formatting.
    
    Args:
        message: The message to print and save
        report_file: File object to save to (optional)
        level: Formatting level - "header", "section", "subsection", or "info"
    """
    # Print to console
    print(message)
    
    # Save to report with proper formatting
    if report_file:
        if level == "header":
            report_file.write(f"# {message}\n\n")
        elif level == "section":
            report_file.write(f"## {message}\n\n")
        elif level == "subsection":
            report_file.write(f"### {message}\n\n")
        else:  # info level
            report_file.write(f"{message}\n\n")
    
    return message

# %% [markdown]
# ## Data Loading Block

# %%
import pandas as pd
import numpy as np
import sys

# Define expected file patterns based on the gene subset file structure
file_patterns = [
    "dataset_type_analysis_v1_rf_config1_genesubsets_shap_seeds20_batch4.pkl",
    "feature_importance_analysis_v1_rf_config1_genesubsets_shap_seeds20_batch4.pkl",
    "shap_consensus_importance_signed_v1_rf_config1_genesubsets_shap_seeds20_batch4.pkl",
    "shap_consensus_importance_v1_rf_config1_genesubsets_shap_seeds20_batch4.pkl",
    "shap_iteration_importance_signed_v1_rf_config1_genesubsets_shap_seeds20_batch4.pkl",
    "shap_iteration_importance_v1_rf_config1_genesubsets_shap_seeds20_batch4.pkl"
]

save_and_print("## Data Loading Results - Gene Subsets", print_report_file, level="section")

# Load datasets with comprehensive error handling
datasets = {}

for pattern in file_patterns:
    file_path = f"{exp_folder}/{pattern}"
    
    # Check if file exists
    if not os.path.exists(file_path):
        save_and_print(f"⚠️  File not found: {file_path}", print_report_file, level="info")
        continue
    
    try:
        # Load the pickle file
        data = pd.read_pickle(file_path)
        dataset_name = pattern.replace(f"_{exp_id}.pkl", "").replace("v1_rf_config1_genesubsets_shap_", "")
        datasets[dataset_name] = data
        
        save_and_print(f"✅ Loaded: {pattern}", print_report_file, level="info")
        
    except Exception as e:
        save_and_print(f"❌ Error loading {pattern}: {str(e)}", print_report_file, level="info")
        save_and_print(f"   Detailed error: {sys.exc_info()[0]}", print_report_file, level="info")

save_and_print(f"📊 Total datasets loaded: {len(datasets)}", print_report_file, level="info")
save_and_print(f"Available datasets: {list(datasets.keys())}", print_report_file, level="info")

# Print RNG seeds used in the experiment
save_and_print("### RNG Seeds Used in Gene Subset Feature Importance Analysis", print_report_file, level="subsection")

if 'feature_importance_analysis' in datasets:
    dataset = datasets['feature_importance_analysis']
    if isinstance(dataset, pd.DataFrame) and 'rng' in dataset.columns:
        rng_seeds = sorted(dataset['rng'].unique())
        save_and_print(f"RNG seeds found: {len(rng_seeds)} unique seeds", print_report_file, level="info")
        save_and_print(f"RNG seed values: {rng_seeds}", print_report_file, level="info")
    else:
        save_and_print("RNG column not found in feature_importance_analysis dataset", print_report_file, level="info")
else:
    save_and_print("feature_importance_analysis dataset not found", print_report_file, level="info")

# %% [markdown]
# ## Section 1: Basic Data Statistics

# %%
import sys

def get_memory_usage(obj):
    """Get memory usage of an object in MB"""
    return sys.getsizeof(obj) / (1024 * 1024)  # Convert to MB

save_and_print("## Basic Statistics - Gene Subsets", print_report_file, level="section")

for name, data in datasets.items():
    save_and_print(f"### Dataset: {name}", print_report_file, level="subsection")
    
    # Shape and basic info
    if hasattr(data, 'shape'):
        save_and_print(f"Shape: {data.shape}", print_report_file, level="info")
        if hasattr(data, 'columns'):
            save_and_print(f"Columns: {len(data.columns)}", print_report_file, level="info")
            if len(data.columns) <= 10:
                save_and_print(f"Column names: {list(data.columns)}", print_report_file, level="info")
            else:
                save_and_print(f"First 10 columns: {list(data.columns[:10])}", print_report_file, level="info")
        if hasattr(data, 'index'):
            if isinstance(data.index, pd.MultiIndex):
                save_and_print(f"Index levels: {data.index.nlevels}", print_report_file, level="info")
            else:
                save_and_print(f"Index length: {len(data.index)}", print_report_file, level="info")
                if len(data.index) <= 10:
                    save_and_print(f"Index names: {list(data.index[:10])}", print_report_file, level="info")
    else:
        save_and_print(f"Type: {type(data)}", print_report_file, level="info")
        if isinstance(data, dict):
            save_and_print(f"Dictionary keys: {len(data)}", print_report_file, level="info")
            if len(data) <= 5:
                save_and_print(f"First keys: {list(data.keys())[:5]}", print_report_file, level="info")
    
    # Memory usage
    memory_mb = get_memory_usage(data)
    save_and_print(f"Memory usage: {memory_mb:.2f} MB", print_report_file, level="info")
    
    # Data types (for DataFrames)
    if isinstance(data, pd.DataFrame):
        save_and_print(f"Data types:", print_report_file, level="info")
        dtype_summary = data.dtypes.value_counts()
        for dtype, count in dtype_summary.items():
            save_and_print(f"  {dtype}: {count} columns", print_report_file, level="info")
    
    # Sample data preview
    if isinstance(data, pd.DataFrame) and len(data) > 0:
        save_and_print(f"First few rows:", print_report_file, level="info")
        save_and_print(data.head(3).to_string(), print_report_file, level="info")
    elif isinstance(data, dict) and len(data) > 0:
        first_key = list(data.keys())[0]
        save_and_print(f"First key sample: {first_key}", print_report_file, level="info")
        if isinstance(data[first_key], pd.Series):
            preview_data = data[first_key].head(3) if len(data[first_key]) > 3 else data[first_key]
            save_and_print(f"First key data: {preview_data.to_string()}", print_report_file, level="info")
        else:
            save_and_print(f"First key value type: {type(data[first_key])}", print_report_file, level="info")

# %% [markdown]
# ## Section 2: Consensus Analysis - Gene Subsets

# %%
save_and_print("## Consensus Analysis - Gene Subsets", print_report_file, level="section")

# Analyze consensus datasets
consensus_datasets = {k: v for k, v in datasets.items() if 'consensus' in k}

if not consensus_datasets:
    save_and_print("No consensus datasets found in loaded data.", print_report_file, level="info")
else:
    for name, data in consensus_datasets.items():
        save_and_print(f"### Consensus Dataset: {name}", print_report_file, level="subsection")
        
        if isinstance(data, pd.DataFrame):
            # Basic metrics
            save_and_print(f"Total features: {len(data)}", print_report_file, level="info")
            
            # Check if this is a consensus importance dataset
            importance_columns = [col for col in data.columns if 'importance' in col.lower() or 'mean' in col.lower()]
            
            if importance_columns:
                # Analyze top features
                if 'mean_importance' in data.columns:
                    top_feature_col = 'mean_importance'
                elif 'importance_score' in data.columns:
                    top_feature_col = 'importance_score'
                else:
                    # Use the first importance-like column
                    top_feature_col = importance_columns[0]
                
                # Get top 10 features
                top_10 = data.nlargest(10, top_feature_col)
                save_and_print(f"Top 10 features by {top_feature_col}:", print_report_file, level="info")
                for i, (idx, row) in enumerate(top_10.iterrows(), 1):
                    importance_val = row[top_feature_col]
                    if 'std' in data.columns:
                        std_val = row.get('std_importance', row.get('std', 0))
                        save_and_print(f"  {i}. {idx}: {importance_val:.4f} ± {std_val:.4f}", print_report_file, level="info")
                    else:
                        save_and_print(f"  {i}. {idx}: {importance_val:.4f}", print_report_file, level="info")
                
                # Stability metrics
                if 'std_importance' in data.columns and 'mean_importance' in data.columns:
                    cv_scores = data['std_importance'] / data['mean_importance']
                    cv_scores = cv_scores.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    save_and_print(f"Stability metrics:", print_report_file, level="info")
                    save_and_print(f"  - Mean CV (std/mean): {cv_scores.mean():.4f}", print_report_file, level="info")
                    save_and_print(f"  - Std of CV: {cv_scores.std():.4f}", print_report_file, level="info")
                    save_and_print(f"  - Min CV: {cv_scores.min():.4f}", print_report_file, level="info")
                    save_and_print(f"  - Max CV: {cv_scores.max():.4f}", print_report_file, level="info")
                
                # Feature importance distribution
                importance_values = data[top_feature_col]
                save_and_print(f"Importance distribution:", print_report_file, level="info")
                save_and_print(f"  - Mean: {importance_values.mean():.4f}", print_report_file, level="info")
                save_and_print(f"  - Std: {importance_values.std():.4f}", print_report_file, level="info")
                save_and_print(f"  - Min: {importance_values.min():.4f}", print_report_file, level="info")
                save_and_print(f"  - Max: {importance_values.max():.4f}", print_report_file, level="info")
                save_and_print(f"  - Median: {importance_values.median():.4f}", print_report_file, level="info")
                
                # Count features with high/low importance
                high_importance = len(importance_values[importance_values > importance_values.quantile(0.75)])
                low_importance = len(importance_values[importance_values < importance_values.quantile(0.25)])
                save_and_print(f"  - High importance (top 25%): {high_importance} features", print_report_file, level="info")
                save_and_print(f"  - Low importance (bottom 25%): {low_importance} features", print_report_file, level="info")
            
        elif isinstance(data, dict):
            # Handle dictionary format consensus data
            save_and_print(f"Dictionary with {len(data)} features", print_report_file, level="info")
            
            # Convert to DataFrame for analysis if possible
            if all(isinstance(v, dict) for v in data.values()):
                try:
                    df_data = []
                    for feature, metrics in data.items():
                        if isinstance(metrics, dict):
                            row = {'feature': feature}
                            row.update(metrics)
                            df_data.append(row)
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('feature', inplace=True)
                        save_and_print(f"Converted to DataFrame with {len(df)} features", print_report_file, level="info")
                        
                        # Analyze similar to DataFrame case
                        importance_cols = [col for col in df.columns if 'importance' in col.lower() or 'mean' in col.lower()]
                        if importance_cols:
                            top_col = importance_cols[0]
                            top_5 = df.nlargest(5, top_col)
                            save_and_print(f"Top 5 features:", print_report_file, level="info")
                            for i, (idx, row) in enumerate(top_5.iterrows(), 1):
                                save_and_print(f"  {i}. {idx}: {row[top_col]:.4f}", print_report_file, level="info")
                except Exception as e:
                    save_and_print(f"Could not convert dictionary to DataFrame: {e}", print_report_file, level="info")

# %% [markdown]
# ## Section 3: SHAP Bidirectional Bar Charts (Individual Conditions) - Gene Subsets

# %%
def parse_condition_title_genesubset(condition):
    """
    Parse technical condition string and generate natural language title for gene subsets.
    
    Args:
        condition: String like "RandomForestRegressor_config1_k500_mrmr_cdk46_genesubset"
    
    Returns:
        Natural language title for the visualization
    """
    parts = condition.split('_')
    
    # Map components to natural language
    model_mapping = {
        'RandomForestRegressor': 'Random Forest',
        'MLPRegressor': 'Multi-layer Perceptron',
        'KNeighborsRegressor': 'K-Neighbors'
    }
    
    network_mapping = {
        'cdk46': 'CDK4/6',
        'fgfr4': 'FGFR4'
    }
    
    dataset_mapping = {
        'dynamic': 'Dynamic Features',
        'genesubset': 'Network-Specific Gene Subsets',  # Changed from 'rnaseq'
        'combined': 'Combined Dataset'
    }
    
    # Extract components (handle variable length parts)
    model = parts[0] if len(parts) > 0 else 'Unknown'
    network = parts[-2] if len(parts) > 1 else 'Unknown'
    dataset_type = parts[-1] if len(parts) > 0 else 'Unknown'
    
    # Apply natural language mapping
    model_name = model_mapping.get(model, model)
    network_name = network_mapping.get(network, network)
    dataset_name = dataset_mapping.get(dataset_type, dataset_type)
    
    # Generate natural title emphasizing gene subsets
    return f"SHAP Analysis: {network_name} Target Model\n{model_name} - {dataset_name}\nDirectional Impact of Network-Specific Gene Subsets"

save_and_print("## SHAP Bidirectional Bar Charts: Gene Subsets - Positive vs Negative Impact", print_report_file, level="section")

# Check if we have signed SHAP data
if 'shap_consensus_importance_signed' in datasets:
    signed_consensus = datasets['shap_consensus_importance_signed']
    
    save_and_print("### Signed SHAP Data Analysis - Gene Subsets", print_report_file, level="subsection")
    
    # Analyze the structure of signed data
    if isinstance(signed_consensus, pd.DataFrame):
        # Extract conditions from multiindex
        conditions = signed_consensus.index.get_level_values(0).unique()
        
        save_and_print(f"Available conditions in signed SHAP data: {list(conditions)}", print_report_file, level="info")
        
        # Create bidirectional bar charts for each condition
        for condition in conditions:
            save_and_print(f"### Analysis for condition: {condition}", print_report_file, level="subsection")
            
            # Extract data for this condition
            condition_data = signed_consensus.xs(condition, level=0, drop_level=False)
            
            if len(condition_data) == 0:
                save_and_print(f"No data found for condition: {condition}", print_report_file, level="info")
                continue
            
            # Analyze directional effects
            if 'mean_importance_signed' in condition_data.columns:
                positive_effects = condition_data[condition_data['mean_importance_signed'] > 0]
                negative_effects = condition_data[condition_data['mean_importance_signed'] < 0]
                neutral_effects = condition_data[condition_data['mean_importance_signed'] == 0]
                
                save_and_print(f"Directional effects:", print_report_file, level="info")
                save_and_print(f"  - Positive effects: {len(positive_effects)} features", print_report_file, level="info")
                save_and_print(f"  - Negative effects: {len(negative_effects)} features", print_report_file, level="info")
                save_and_print(f"  - Neutral effects: {len(neutral_effects)} features", print_report_file, level="info")
                
                # Create bidirectional bar chart
                try:
                    # Set up publication-quality styling
                    plt.style.use('seaborn-v0_8')
                    plt.rcParams['font.family'] = 'sans-serif'
                    plt.rcParams['font.size'] = 14
                    plt.rcParams['axes.linewidth'] = 1.2
                    
                    # Take top 20 features by absolute signed importance (both positive and negative)
                    condition_data_copy = condition_data.copy()
                    condition_data_copy['abs_importance'] = condition_data_copy['mean_importance_signed'].abs()
                    
                    # Get top 20 features by absolute importance
                    top_features = condition_data_copy.nlargest(20, 'abs_importance')
                    
                    # Sort by signed value for better visualization (negative to positive)
                    top_features = top_features.sort_values('mean_importance_signed', ascending=True)
                    
                    # Create horizontal bar plot with color coding for direction
                    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
                    
                    # Create color mapping based on sign
                    colors = ['red' if x < 0 else 'blue' for x in top_features['mean_importance_signed']]
                    
                    # Create the bar plot
                    bars = ax.barh(range(len(top_features)), top_features['mean_importance_signed'], 
                                  color=colors, alpha=0.7, edgecolor='black', linewidth=1)
                    
                    # Set y-axis labels to feature names
                    y_positions = range(len(top_features))
                    feature_names = [str(feature) for feature in top_features.index.get_level_values(1)]
                    ax.set_yticks(y_positions)
                    ax.set_yticklabels(feature_names, fontsize=12)
                    
                    # Label axes
                    ax.set_xlabel('Mean Signed SHAP Value', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Feature', fontsize=14, fontweight='bold')
                    # Generate natural title from condition (using gene subset version)
                    natural_title = parse_condition_title_genesubset(condition)
                    ax.set_title(f'SHAP Directional Feature Effects\n{natural_title}\n(Top 20 Features by Absolute Importance)', 
                                fontsize=16, fontweight='bold', pad=20)

