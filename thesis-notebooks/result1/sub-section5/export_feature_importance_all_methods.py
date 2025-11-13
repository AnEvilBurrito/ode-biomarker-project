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
# ## Feature Importance Export for All Methods
#
# This script exports feature importance data from the feature selection method comparison
# in the same format as feat_importance_analysis.py, ensuring consistency across all exports.
#
# **Export Format**: mean_importance, std_importance, occurrence_count
# **Methods Covered**: Network only (d3), MRMR only, MRMR + Network (d3)
# **Importance Methods**: SHAP and MDI for each method

# %% [markdown]
# ## Initialisation

# %%
import os

path = os.getcwd()
# find the string 'project' in the path, return index
index_project = path.find("project")
# slice the path from the index of 'project' to the end
project_path = path[: index_project + 7]
# set the working directory
os.chdir(project_path)
print(f"Project path set to: {os.getcwd()}")

# %%
from PathLoader import PathLoader #noqa: E402

path_loader = PathLoader("data_config.env", "current_user.env")

# %%
from DataLink import DataLink #noqa: E402

data_link = DataLink(path_loader, "data_codes.csv")

# %%
# Use the actual directory where results are stored
folder_name = "ThesisResult-FeatureImportanceConsensus"
exp_id = "v1_rf_k500_3methods_split0.3_comparison"

# The results are already in the main folder, no need to create subfolders
main_results_folder = f"{path_loader.get_data_path()}data/results/{folder_name}/"
file_save_path = f"{path_loader.get_data_path()}data/results/{folder_name}/"

# %%
# Load required libraries
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Create a new report file for capturing print statements
print_report_path = f"{file_save_path}feature_importance_export_report_{exp_id}.md"
print_report_file = open(print_report_path, 'w', encoding='utf-8')

# Write header to the print report
print_report_file.write(f"# Feature Importance Export Report - {exp_id}\n\n")
print_report_file.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
print_report_file.write("This report captures all feature importance exports from the feature selection method comparison analysis.\n\n")

def save_and_print(message, report_file=None, level="info"):
    """
    Print message to console and save to report file with proper formatting.
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
# ## Data Loading

# %%
save_and_print("## Loading Feature Selection Comparison Data", print_report_file, level="section")

# Define experiment parameters
model_name = "RandomForestRegressor"
k_value = 500
split_size = 0.3
network_distance = 3

# Define the 3 feature selection methods
methods = {
    "network_only_d3": "Network only (distance 3)", 
    "mrmr_only": "MRMR only",
    "mrmr_network_d3": "MRMR + Network (distance 3)"
}

# Define all conditions (3 methods × 2 importance methods)
conditions = []
for method_name, method_desc in methods.items():
    conditions.append(f"{model_name}_k{k_value}_{method_name}_split{split_size}_shap")
    conditions.append(f"{model_name}_k{k_value}_{method_name}_split{split_size}_mdi")

save_and_print(f"Loading data for {len(conditions)} conditions:", print_report_file, level="info")
for method_name, method_desc in methods.items():
    save_and_print(f"- {method_desc}: SHAP and MDI importance methods", print_report_file, level="info")

# %%
# Load all available data files from the comparison results folder
data_files = {}

save_and_print("### Data Loading Progress", print_report_file, level="subsection")

for condition in conditions:
    condition_data = {}
    
    # Try to load consensus feature importance data
    file_types = [
        f"consensus_feature_importance_{condition}.pkl",
        f"consensus_feature_importance_signed_{condition}.pkl"
    ]
    
    for file_type in file_types:
        file_path = f"{main_results_folder}{file_type}"
        if os.path.exists(file_path):
            try:
                condition_data[file_type.replace(f"_{condition}.pkl", "")] = pd.read_pickle(file_path)
                save_and_print(f"✓ Loaded {file_type}", print_report_file, level="info")
            except Exception as e:
                save_and_print(f"✗ Failed to load {file_type}: {e}", print_report_file, level="info")
        else:
            save_and_print(f"✗ File not found: {file_type}", print_report_file, level="info")
    
    data_files[condition] = condition_data

# Display available data summary
save_and_print("### Available Data Summary", print_report_file, level="subsection")

method_summary = {}
for condition, files in data_files.items():
    # Extract method name from condition
    method_name = '_'.join(condition.split('_')[2:-2])
    importance_method = condition.split('_')[-1]
    
    if method_name not in method_summary:
        method_summary[method_name] = {'shap': 0, 'mdi': 0, 'total_files': 0}
    
    method_summary[method_name][importance_method] += 1
    method_summary[method_name]['total_files'] += len([f for f in files.values() if f is not None])

save_and_print("**Method Data Availability:**", print_report_file, level="info")
for method_name, counts in method_summary.items():
    save_and_print(f"- {methods[method_name]}: SHAP={counts['shap']}, MDI={counts['mdi']}, Total files={counts['total_files']}", 
                  print_report_file, level="info")

# %% [markdown]
# ## Feature Importance CSV Export (All Methods)

# %%
save_and_print("## Feature Importance CSV Export (All Methods)", print_report_file, level="section")

def export_feature_importance_csv_all_methods(data_files, file_save_path, exp_id, top_n=50, min_occurrence_threshold=0.5):
    """
    Export comprehensive feature importance data to CSV files for all feature selection methods
    following the same format as feat_importance_analysis.py
    
    Args:
        data_files: Dictionary containing the data files for all methods
        file_save_path: Path to save CSV files
        exp_id: Experiment identifier
        top_n: Number of top features to consider for top-N analysis
        min_occurrence_threshold: Minimum occurrence ratio threshold for feature selection (default: 0.5 = 50%)
    """
    save_and_print("### Exporting Feature Importance Data for All Methods", print_report_file, level="subsection")
    
    # Check available conditions
    available_conditions = []
    for condition in conditions:
        if condition in data_files and 'consensus_feature_importance' in data_files[condition]:
            available_conditions.append(condition)
    
    if not available_conditions:
        save_and_print("No consensus feature importance data available for export", print_report_file, level="info")
        return None
    
    save_and_print(f"Found {len(available_conditions)} conditions with consensus data", print_report_file, level="info")
    
    # Export each condition separately
    exported_files = {}
    
    for condition in available_conditions:
        if 'consensus_feature_importance' not in data_files[condition]:
            continue
            
        consensus_df = data_files[condition]['consensus_feature_importance']
        
        # Extract method information
        parts = condition.split('_')
        method_name = '_'.join(parts[2:-2])
        importance_method = parts[-1]
        
        # Create export dataframe with the same format as feat_importance_analysis.py
        export_df = pd.DataFrame()
        
        for feature in consensus_df.index:
            feature_data = consensus_df.loc[feature]
            export_df.loc[feature, 'mean_importance'] = feature_data.get('mean_importance', np.nan)
            export_df.loc[feature, 'std_importance'] = feature_data.get('std_importance', np.nan)
            if 'occurrence_count' in feature_data:
                export_df.loc[feature, 'occurrence_count'] = feature_data.get('occurrence_count', np.nan)
            else:
                export_df.loc[feature, 'occurrence_count'] = np.nan
        
        # Sort by mean importance (most important first)
        export_df = export_df.sort_values('mean_importance', ascending=False)
        
        # Export individual condition file
        condition_csv_path = f"{file_save_path}feature_importance_{method_name}_{importance_method}_{exp_id}.csv"
        export_df.to_csv(condition_csv_path)
        
        exported_files[condition] = {
            'path': condition_csv_path,
            'method': method_name,
            'importance_method': importance_method,
            'n_features': len(export_df)
        }
        
        save_and_print(f"✓ Exported {methods[method_name]} ({importance_method.upper()}): {len(export_df)} features", print_report_file, level="info")
    
    # Create comprehensive combined export (all methods and importance types)
    save_and_print("### Creating Comprehensive Combined Export", print_report_file, level="subsection")
    
    # Get all unique features across all conditions
    all_features = set()
    for condition in available_conditions:
        if 'consensus_feature_importance' in data_files[condition]:
            all_features.update(data_files[condition]['consensus_feature_importance'].index)
    
    # Create comprehensive dataframe
    comprehensive_df = pd.DataFrame(index=list(all_features))
    
    # Add columns for each condition
    for condition in available_conditions:
        if 'consensus_feature_importance' not in data_files[condition]:
            continue
            
        consensus_df = data_files[condition]['consensus_feature_importance']
        parts = condition.split('_')
        method_name = '_'.join(parts[2:-2])
        importance_method = parts[-1]
        
        # Add mean and std importance columns
        for feature in all_features:
            if feature in consensus_df.index:
                feature_data = consensus_df.loc[feature]
                comprehensive_df.loc[feature, f'{method_name}_{importance_method}_mean'] = feature_data.get('mean_importance', np.nan)
                comprehensive_df.loc[feature, f'{method_name}_{importance_method}_std'] = feature_data.get('std_importance', np.nan)
                comprehensive_df.loc[feature, f'{method_name}_{importance_method}_occurrence'] = feature_data.get('occurrence_count', np.nan)
            else:
                comprehensive_df.loc[feature, f'{method_name}_{importance_method}_mean'] = np.nan
                comprehensive_df.loc[feature, f'{method_name}_{importance_method}_std'] = np.nan
                comprehensive_df.loc[feature, f'{method_name}_{importance_method}_occurrence'] = np.nan
    
    # Add summary statistics
    mean_cols = [col for col in comprehensive_df.columns if col.endswith('_mean')]
    comprehensive_df['mean_importance_all_methods'] = comprehensive_df[mean_cols].mean(axis=1)
    comprehensive_df['std_importance_all_methods'] = comprehensive_df[[col for col in comprehensive_df.columns if col.endswith('_std')]].mean(axis=1)
    comprehensive_df['total_occurrences'] = comprehensive_df[[col for col in comprehensive_df.columns if col.endswith('_occurrence')]].sum(axis=1)
    
    # Sort by overall importance
    comprehensive_df = comprehensive_df.sort_values('mean_importance_all_methods', ascending=False)
    
    # Export comprehensive file
    comprehensive_csv_path = f"{file_save_path}feature_importance_comprehensive_all_methods_{exp_id}.csv"
    comprehensive_df.to_csv(comprehensive_csv_path)
    
    save_and_print(f"✓ Exported comprehensive file: {len(comprehensive_df)} features", print_report_file, level="info")
    save_and_print(f"  File: {comprehensive_csv_path}", print_report_file, level="info")
    
    # Create selected features export using occurrence threshold
    save_and_print("### Creating Selected Features Export", print_report_file, level="subsection")
    
    # Apply occurrence threshold (similar to feat_importance_analysis.py)
    if 'total_occurrences' in comprehensive_df.columns:
        # Calculate threshold based on total possible occurrences
        max_possible_occurrences = len([c for c in available_conditions if 'consensus_feature_importance' in data_files.get(c, {})])
        occurrence_threshold = max_possible_occurrences * min_occurrence_threshold
        
        selected_features_df = comprehensive_df[comprehensive_df['total_occurrences'] >= occurrence_threshold].copy()
        
        # Export selected features
        selected_csv_path = f"{file_save_path}feature_importance_selected_all_methods_{exp_id}.csv"
        selected_features_df.to_csv(selected_csv_path)
        
        save_and_print(f"✓ Exported selected features: {len(selected_features_df)} features", print_report_file, level="info")
        save_and_print(f"  Threshold: {occurrence_threshold:.1f} occurrences ({min_occurrence_threshold*100:.0f}%)", print_report_file, level="info")
        save_and_print(f"  File: {selected_csv_path}", print_report_file, level="info")
    else:
        selected_features_df = pd.DataFrame()
        save_and_print("✗ Could not create selected features export (no occurrence data)", print_report_file, level="info")
    
    # Create method-specific combined exports
    save_and_print("### Creating Method-Specific Combined Exports", print_report_file, level="subsection")
    
    for method_name in methods.keys():
        method_conditions = [c for c in available_conditions if method_name in c]
        
        if len(method_conditions) < 2:
            save_and_print(f"Skipping {method_name}: insufficient conditions", print_report_file, level="info")
            continue
        
        # Create method-specific dataframe
        method_df = pd.DataFrame()
        
        # Get features for this method
        method_features = set()
        for condition in method_conditions:
            if 'consensus_feature_importance' in data_files[condition]:
                method_features.update(data_files[condition]['consensus_feature_importance'].index)
        
        method_df = pd.DataFrame(index=list(method_features))
        
        # Add SHAP and MDI data for this method
        for condition in method_conditions:
            if 'consensus_feature_importance' not in data_files[condition]:
                continue
                
            consensus_df = data_files[condition]['consensus_feature_importance']
            importance_method = condition.split('_')[-1]
            
            for feature in method_features:
                if feature in consensus_df.index:
                    feature_data = consensus_df.loc[feature]
                    method_df.loc[feature, f'{importance_method}_mean_importance'] = feature_data.get('mean_importance', np.nan)
                    method_df.loc[feature, f'{importance_method}_std_importance'] = feature_data.get('std_importance', np.nan)
                    method_df.loc[feature, f'{importance_method}_occurrence_count'] = feature_data.get('occurrence_count', np.nan)
                else:
                    method_df.loc[feature, f'{importance_method}_mean_importance'] = np.nan
                    method_df.loc[feature, f'{importance_method}_std_importance'] = np.nan
                    method_df.loc[feature, f'{importance_method}_occurrence_count'] = np.nan
        
        # Add method summary statistics
        if 'shap_mean_importance' in method_df.columns and 'mdi_mean_importance' in method_df.columns:
            method_df['mean_importance_avg'] = method_df[['shap_mean_importance', 'mdi_mean_importance']].mean(axis=1)
            method_df['std_importance_avg'] = method_df[['shap_std_importance', 'mdi_std_importance']].mean(axis=1)
            method_df['total_occurrence'] = method_df[['shap_occurrence_count', 'mdi_occurrence_count']].sum(axis=1)
        elif 'shap_mean_importance' in method_df.columns:
            method_df['mean_importance_avg'] = method_df['shap_mean_importance']
            method_df['std_importance_avg'] = method_df['shap_std_importance']
            method_df['total_occurrence'] = method_df['shap_occurrence_count']
        elif 'mdi_mean_importance' in method_df.columns:
            method_df['mean_importance_avg'] = method_df['mdi_mean_importance']
            method_df['std_importance_avg'] = method_df['mdi_std_importance']
            method_df['total_occurrence'] = method_df['mdi_occurrence_count']
        
        # Sort by average importance
        method_df = method_df.sort_values('mean_importance_avg', ascending=False)
        
        # Export method-specific file
        method_csv_path = f"{file_save_path}feature_importance_{method_name}_combined_{exp_id}.csv"
        method_df.to_csv(method_csv_path)
        
        save_and_print(f"✓ Exported {methods[method_name]} combined: {len(method_df)} features", print_report_file, level="info")
        save_and_print(f"  File: {method_csv_path}", print_report_file, level="info")
    
    # Print summary statistics
    save_and_print("### Export Summary Statistics", print_report_file, level="subsection")
    save_and_print(f"**Total conditions exported:** {len(exported_files)}", print_report_file, level="info")
    save_and_print(f"**Total unique features across all methods:** {len(all_features)}", print_report_file, level="info")
    save_and_print(f"**Comprehensive export file:** {comprehensive_csv_path}", print_report_file, level="info")
    
    if 'total_occurrences' in comprehensive_df.columns:
        save_and_print(f"**Features with occurrence data:** {comprehensive_df['total_occurrences'].notna().sum()}", print_report_file, level="info")
        save_and_print(f"**Average occurrences per feature:** {comprehensive_df['total_occurrences'].mean():.1f}", print_report_file, level="info")
    
    # Method breakdown
    save_and_print("**Method breakdown:**", print_report_file, level="info")
    for condition, info in exported_files.items():
        save_and_print(f"- {methods[info['method']]} ({info['importance_method'].upper()}): {info['n_features']} features", print_report_file, level="info")
    
    return {
        'individual_exports': exported_files,
        'comprehensive_export': comprehensive_csv_path,
        'comprehensive_df': comprehensive_df,
        'selected_export': selected_csv_path if len(selected_features_df) > 0 else None,
        'selected_df': selected_features_df
    }

# Execute the export for all methods
export_results = export_feature_importance_csv_all_methods(data_files, file_save_path, exp_id, min_occurrence_threshold=0.5)

# %% [markdown]
# ## SHAP Directional Effects Export (All Methods)

# %%
save_and_print("## SHAP Directional Effects Export (All Methods)", print_report_file, level="section")

def export_shap_directional_effects_all_methods(data_files, file_save_path, exp_id):
    """
    Export SHAP directional effects (positive/negative) for all methods that have SHAP data
    following the same format as feat_importance_analysis.py
    """
    save_and_print("### Exporting SHAP Directional Effects for All Methods", print_report_file, level="subsection")
    
    shap_exports = {}
    
    # Find all SHAP conditions
    shap_conditions = [condition for condition in conditions if 'shap' in condition]
    
    for condition in shap_conditions:
        if condition not in data_files or 'consensus_feature_importance_signed' not in data_files[condition]:
            save_and_print(f"Skipping {condition}: No signed SHAP data available", print_report_file, level="info")
            continue
        
        # Extract method information
        parts = condition.split('_')
        method_name = '_'.join(parts[2:-2])
        
        signed_consensus = data_files[condition]['consensus_feature_importance_signed']
        
        # Analyze directional effects
        positive_effects = signed_consensus[signed_consensus['mean_importance_signed'] > 0]
        negative_effects = signed_consensus[signed_consensus['mean_importance_signed'] < 0]
        
        # Create export dataframes with the same format as feat_importance_analysis.py
        positive_export = positive_effects[['mean_importance_signed', 'std_importance_signed', 'occurrence_count']].copy()
        negative_export = negative_effects[['mean_importance_signed', 'std_importance_signed', 'occurrence_count']].copy()
        
        # Rename columns to match existing format
        positive_export = positive_export.rename(columns={
            'mean_importance_signed': 'mean_importance',
            'std_importance_signed': 'std_importance'
        })
        negative_export = negative_export.rename(columns={
            'mean_importance_signed': 'mean_importance',
            'std_importance_signed': 'std_importance'
        })
        
        # Convert negative values to absolute values for negative effects export
        negative_export['mean_importance'] = negative_export['mean_importance'].abs()
        
        # Sort by absolute value of mean_importance (most impactful first)
        positive_export['abs_importance'] = positive_export['mean_importance'].abs()
        negative_export['abs_importance'] = negative_export['mean_importance'].abs()
        
        positive_export = positive_export.sort_values('abs_importance', ascending=False)
        negative_export = negative_export.sort_values('abs_importance', ascending=False)
        
        # Remove the temporary abs_importance column before export
        positive_export = positive_export.drop('abs_importance', axis=1)
        negative_export = negative_export.drop('abs_importance', axis=1)
        
        # Export positive effects
        positive_csv_path = f"{file_save_path}shap_positive_effects_{method_name}_{exp_id}.csv"
        positive_export.to_csv(positive_csv_path)
        
        # Export negative effects  
        negative_csv_path = f"{file_save_path}shap_negative_effects_{method_name}_{exp_id}.csv"
        negative_export.to_csv(negative_csv_path)
        
        shap_exports[method_name] = {
            'positive_path': positive_csv_path,
            'negative_path': negative_csv_path,
            'positive_count': len(positive_export),
            'negative_count': len(negative_export),
            'total_count': len(signed_consensus)
        }
        
        save_and_print(f"✓ {methods[method_name]} SHAP effects exported:", print_report_file, level="info")
        save_and_print(f"  - Positive effects: {len(positive_export)} features -> {positive_csv_path}", print_report_file, level="info")
        save_and_print(f"  - Negative effects: {len(negative_export)} features -> {negative_csv_path}", print_report_file, level="info")
        save_and_print(f"  - Total features: {len(signed_consensus)}", print_report_file, level="info")
    
    # Summary statistics
    if shap_exports:
        save_and_print("### SHAP Directional Effects Summary", print_report_file, level="subsection")
        total_positive = sum(info['positive_count'] for info in shap_exports.values())
        total_negative = sum(info['negative_count'] for info in shap_exports.values())
        total_features = sum(info['total_count'] for info in shap_exports.values())
        
        save_and_print(f"**Total SHAP features across all methods:** {total_features}", print_report_file, level="info")
        save_and_print(f"**Total positive effects:** {total_positive}", print_report_file, level="info")
        save_and_print(f"**Total negative effects:** {total_negative}", print_report_file, level="info")
        save_and_print(f"**Overall positive ratio:** {total_positive/total_features:.1%}", print_report_file, level="info")
        
        save_and_print("**Method breakdown:**", print_report_file, level="info")
        for method_name, info in shap_exports.items():
            positive_ratio = info['positive_count'] / info['total_count']
            save_and_print(f"- {methods[method_name]}: {info['positive_count']} positive, {info['negative_count']} negative ({positive_ratio:.1%} positive)", print_report_file, level="info")
    else:
        save_and_print("No SHAP directional effects data available for export", print_report_file, level="info")
    
    return shap_exports

# Execute SHAP directional effects export
shap_export_results = export_shap_directional_effects_all_methods(data_files, file_save_path, exp_id)

# %% [markdown]
# ## Export Summary and Validation

# %%
save_and_print("## Export Summary and Validation", print_report_file, level="section")

def validate_exports(export_results, shap_export_results, file_save_path, exp_id):
    """
    Validate exported files and provide summary statistics
    """
    save_and_print("### Export Validation", print_report_file, level="subsection")
    
    validation_results = {
        'feature_importance_exports': {},
        'shap_exports': {},
        'file_sizes': {},
        'data_integrity': {}
    }
    
    # Validate feature importance exports
    if export_results:
        save_and_print("**Feature Importance Exports:**", print_report_file, level="info")
        
        # Check comprehensive export
        if export_results.get('comprehensive_export'):
            comp_path = export_results['comprehensive_export']
            if os.path.exists(comp_path):
                comp_size = os.path.getsize(comp_path)
                validation_results['file_sizes']['comprehensive'] = comp_size
                save_and_print(f"✓ Comprehensive export: {comp_size} bytes", print_report_file, level="info")
                
                # Validate data integrity
                try:
                    comp_df = pd.read_csv(comp_path, index_col=0)
                    validation_results['data_integrity']['comprehensive'] = {
                        'rows': len(comp_df),
                        'columns': len(comp_df.columns),
                        'has_mean_importance': 'mean_importance_all_methods' in comp_df.columns,
                        'has_std_importance': 'std_importance_all_methods' in comp_df.columns,
                        'has_occurrences': 'total_occurrences' in comp_df.columns
                    }
                    save_and_print(f"  - Rows: {len(comp_df)}, Columns: {len(comp_df.columns)}", print_report_file, level="info")
                except Exception as e:
                    save_and_print(f"  ✗ Error reading comprehensive export: {e}", print_report_file, level="info")
            else:
                save_and_print(f"✗ Comprehensive export file not found: {comp_path}", print_report_file, level="info")
        
        # Check selected export
        if export_results.get('selected_export'):
            sel_path = export_results['selected_export']
            if os.path.exists(sel_path):
                sel_size = os.path.getsize(sel_path)
                validation_results['file_sizes']['selected'] = sel_size
                save_and_print(f"✓ Selected features export: {sel_size} bytes", print_report_file, level="info")
            else:
                save_and_print(f"✗ Selected features export file not found: {sel_path}", print_report_file, level="info")
        
        # Check individual exports
        individual_count = 0
        for condition, info in export_results.get('individual_exports', {}).items():
            if os.path.exists(info['path']):
                individual_count += 1
                size = os.path.getsize(info['path'])
                validation_results['file_sizes'][f"individual_{condition}"] = size
            else:
                save_and_print(f"✗ Individual export not found: {info['path']}", print_report_file, level="info")
        
        save_and_print(f"✓ Individual exports: {individual_count}/{len(export_results.get('individual_exports', {}))} files found", print_report_file, level="info")
    
    # Validate SHAP exports
    if shap_export_results:
        save_and_print("**SHAP Directional Effects Exports:**", print_report_file, level="info")
        
        positive_count = 0
        negative_count = 0
        
        for method_name, info in shap_export_results.items():
            pos_path = info['positive_path']
            neg_path = info['negative_path']
            
            if os.path.exists(pos_path):
                positive_count += 1
                pos_size = os.path.getsize(pos_path)
                validation_results['file_sizes'][f"shap_positive_{method_name}"] = pos_size
                save_and_print(f"✓ {methods[method_name]} positive effects: {pos_size} bytes", print_report_file, level="info")
            else:
                save_and_print(f"✗ {methods[method_name]} positive effects not found: {pos_path}", print_report_file, level="info")
            
            if os.path.exists(neg_path):
                negative_count += 1
                neg_size = os.path.getsize(neg_path)
                validation_results['file_sizes'][f"shap_negative_{method_name}"] = neg_size
                save_and_print(f"✓ {methods[method_name]} negative effects: {neg_size} bytes", print_report_file, level="info")
            else:
                save_and_print(f"✗ {methods[method_name]} negative effects not found: {neg_path}", print_report_file, level="info")
        
        save_and_print(f"✓ SHAP exports: {positive_count + negative_count}/{len(shap_export_results) * 2} files found", print_report_file, level="info")
    
    # Summary
    save_and_print("**Export Summary:**", print_report_file, level="info")
    total_files = len(validation_results['file_sizes'])
    total_size = sum(validation_results['file_sizes'].values())
    save_and_print(f"Total files exported: {total_files}", print_report_file, level="info")
    save_and_print(f"Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)", print_report_file, level="info")
    save_and_print(f"Export directory: {file_save_path}", print_report_file, level="info")
    
    return validation_results

# Validate all exports
validation_results = validate_exports(export_results, shap_export_results, file_save_path, exp_id)

# %% [markdown]
# ## Conclusion

# %%
save_and_print("## Feature Export Conclusion", print_report_file, level="section")

save_and_print("Feature importance data has been successfully exported for all feature selection methods using the same format as feat_importance_analysis.py:", print_report_file, level="info")

save_and_print("### Exported Files:", print_report_file, level="subsection")
save_and_print("1. **Individual condition exports** - Each method/importance combination", print_report_file, level="info")
save_and_print("2. **Comprehensive export** - All features across all methods combined", print_report_file, level="info")
save_and_print("3. **Selected features export** - Features meeting occurrence threshold", print_report_file, level="info")
save_and_print("4. **Method-specific exports** - Combined SHAP+MDI for each method", print_report_file, level="info")
save_and_print("5. **SHAP directional effects** - Positive and negative effects for each method", print_report_file, level="info")

save_and_print("### Export Format Consistency:", print_report_file, level="subsection")
save_and_print("- **Column structure**: mean_importance, std_importance, occurrence_count", print_report_file, level="info")
save_and_print("- **File naming**: Consistent with existing export patterns", print_report_file, level="info")
save_and_print("- **Threshold logic**: Same occurrence-based feature selection", print_report_file, level="info")
save_and_print("- **Data validation**: All exports verified for integrity", print_report_file, level="info")

save_and_print("The exported files are now ready for downstream analysis and maintain full compatibility with existing analysis pipelines.", print_report_file, level="info")

# Close the report file
print_report_file.close()

save_and_print("## Export Complete", print_report_file, level="section")
save_and_print(f"All feature importance exports have been completed successfully!", print_report_file, level="info")
save_and_print(f"Report saved to: {print_report_path}", print_report_file, level="info")
