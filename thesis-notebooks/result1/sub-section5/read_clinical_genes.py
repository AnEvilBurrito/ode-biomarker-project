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
# ## Feature Importance Analysis Notebook
#
# This notebook performs comprehensive analysis of feature importance results from the consensus analysis, including:
# - Jaccard stability similarity analysis comparing SHAP, MDI, and feature selection scores
# - Convergence analysis with AUC calculation for tolerance drop curves
# - SHAP signed values analysis for directional effects

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
from PathLoader import PathLoader  # noqa: E402

path_loader = PathLoader("data_config.env", "current_user.env")

# %%
from DataLink import DataLink  # noqa: E402

data_link = DataLink(path_loader, "data_codes.csv")

# %%
folder_name = "ThesisResult-FeatureImportanceConsensus"
exp_id = "v2_rf_k500_network_d3_split0.3"  # Without _importance_consensus suffix

# Create both the main folder and exp_id subfolder
main_folder = f"{path_loader.get_data_path()}data/results/{folder_name}"
exp_folder = f"{main_folder}/{exp_id}"

if not os.path.exists(main_folder):
    os.makedirs(main_folder)
if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

file_save_path = f"{exp_folder}/"



# %%
import pandas as pd
import numpy as np

clinical_genes_df = pd.read_excel("gene_to_uniprot_mapping.xlsx", sheet_name="Sheet1")

clinical_genes_df.head()

# %%
# Load proteomics data 
print("## Data Loading and Preparation")
print("Loading proteomics data...")
loading_code = "goncalves-gdsc-2-Palbociclib-LN_IC50-sin"
proteomic_feature_data, proteomic_label_data = data_link.get_data_using_code(loading_code)

print(f"Proteomic feature data shape: {proteomic_feature_data.shape}")
print(f"Proteomic label data shape: {proteomic_label_data.shape}")

print("Preparing and aligning data...")
proteomic_feature_data = proteomic_feature_data.select_dtypes(include=[np.number])

# Align indices
common_indices = sorted(
    set(proteomic_feature_data.index) & set(proteomic_label_data.index)
)
feature_data = proteomic_feature_data.loc[common_indices]
label_data = proteomic_label_data.loc[common_indices]

print(f"Final aligned dataset shape: {feature_data.shape}")
print(f"Final aligned label shape: {label_data.shape}")

# %% [markdown]
# ## Cross-Matching Clinical Genes with Proteomics Features

# %%
# Examine proteomics feature names to understand the format
print("\n## Examining Proteomics Feature Names")
print("Sample feature names:")
for i, col in enumerate(feature_data.columns[:10]):
    print(f"{i+1}. {col}")

print(f"\nTotal number of proteomics features: {len(feature_data.columns)}")

# Extract Uniprot IDs from proteomics feature names
def extract_uniprot_id(feature_name):
    """
    Extract Uniprot ID from proteomics feature name format: [Gene][UniprotID]:HUMAN
    Example: 'EGFRP00533:HUMAN' -> 'P00533'
    """
    # Look for pattern: any characters followed by P followed by 5 digits
    import re
    match = re.search(r'([A-Z][0-9]{5})', feature_name)
    if match:
        return match.group(1)
    return None

# Extract Uniprot IDs from all feature names
feature_uniprot_ids = {}
for col in feature_data.columns:
    uniprot_id = extract_uniprot_id(col)
    if uniprot_id:
        feature_uniprot_ids[col] = uniprot_id

print(f"\nNumber of features with extractable Uniprot IDs: {len(feature_uniprot_ids)}")

# Get clinical gene Uniprot IDs
clinical_uniprot_ids = set(clinical_genes_df['UniProt ID'].tolist())
print(f"Number of clinical genes: {len(clinical_uniprot_ids)}")

# Find overlapping Uniprot IDs
overlapping_ids = clinical_uniprot_ids.intersection(set(feature_uniprot_ids.values()))
print(f"Number of overlapping Uniprot IDs: {len(overlapping_ids)}")

# Create mapping of clinical genes to proteomics features
clinical_to_proteomics = {}
for clinical_id in overlapping_ids:
    matching_features = [feature for feature, uniprot_id in feature_uniprot_ids.items() 
                        if uniprot_id == clinical_id]
    clinical_to_proteomics[clinical_id] = matching_features

print("\n## Matching Results")
print(f"Clinical genes found in proteomics dataset: {len(overlapping_ids)}")
print("Matching Uniprot IDs:")
for uniprot_id in sorted(overlapping_ids):
    gene_symbol = clinical_genes_df[clinical_genes_df['UniProt ID'] == uniprot_id]['Gene Symbol'].iloc[0]
    matching_features = clinical_to_proteomics[uniprot_id]
    print(f"  {gene_symbol} ({uniprot_id}): {len(matching_features)} feature(s)")
    for feature in matching_features:
        print(f"    - {feature}")

# Create filtered dataset with only clinically relevant features
clinical_feature_columns = []
for features_list in clinical_to_proteomics.values():
    clinical_feature_columns.extend(features_list)

clinical_feature_data = feature_data[clinical_feature_columns]
print(f"\nFiltered dataset shape (clinical features only): {clinical_feature_data.shape}")
