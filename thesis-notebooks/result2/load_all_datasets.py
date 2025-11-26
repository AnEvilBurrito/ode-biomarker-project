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
folder_name = "ThesisResult2-1-DynamicFeatures"
exp_id = "v1"

if not os.path.exists(f"{path_loader.get_data_path()}data/results/{folder_name}/{exp_id}"):
    os.makedirs(f"{path_loader.get_data_path()}data/results/{folder_name}/{exp_id}")

file_save_path = f"{path_loader.get_data_path()}data/results/{folder_name}/{exp_id}/"


# %% [markdown]
# ### Loading CDK4/6 RNASeq data and Dynamic Features

# %%
# Load Transcriptomics Palbociclib dataset
loading_code = "ccle-gdsc-2-Palbociclib-LN_IC50"
cdk46_rnaseq_feature_data, cdk46_rnaseq_label_data = data_link.get_data_using_code(
    loading_code
)

print(f"RNASeq feature data shape: {cdk46_rnaseq_feature_data.shape}")
print(f"RNASeq label data shape: {cdk46_rnaseq_label_data.shape}")

# %%

feature_data_dynamic, label_data_dynamic = data_link.get_data_using_code('generic-gdsc-2-Palbociclib-LN_IC50-cdk46_ccle_dynamic_features_v4_ccle-true-Unnamed: 0')
print(f"Dynamic dataset shape: {feature_data_dynamic.shape}")
print(f"Dynamic label shape: {label_data_dynamic.shape}")

# %% [markdown]
# ### Loading CDK4/6 Proteomics data and Dynamic Features


# %%
# Load Proteomics Palbociclib dataset
loading_code = "goncalves-gdsc-2-Palbociclib-LN_IC50-sin"
proteomic_feature_data, proteomic_label_data = data_link.get_data_using_code(
    loading_code
)

print(f"Proteomic feature data shape: {proteomic_feature_data.shape}")
print(f"Proteomic label data shape: {proteomic_label_data.shape}")

# %%

feature_data_dynamic, label_data_dynamic = data_link.get_data_using_code('generic-gdsc-2-Palbociclib-LN_IC50-cdk46_ccle_dynamic_features_v4_ccle_proteomics-true-Unnamed: 0')
print(f"Proteomic Dynamic dataset shape: {feature_data_dynamic.shape}")
print(f"Proteomic Dynamic label shape: {label_data_dynamic.shape}")


# %% [markdown]
# ### Loading FGFR4 RNASeq data and Dynamic Features


# %%
# Load FGFR4 RNASeq dataset
loading_code = "ccle-gdsc-1-FGFR_0939-LN_IC50"
fgfr4_RNASeq_feature_data, fgfr4_RNASeq_label_data = data_link.get_data_using_code(
    loading_code
)

print(f"feature data shape: {fgfr4_RNASeq_feature_data.shape}")
print(f"label data shape: {fgfr4_RNASeq_label_data.shape}")

# %%

feature_data_dynamic, label_data_dynamic = data_link.get_data_using_code('generic-gdsc-1-FGFR_0939-LN_IC50-fgfr4_ccle_dynamic_features_v2-true-Unnamed: 0')
print(f"Dynamic dataset shape: {feature_data_dynamic.shape}")
print(f"Dynamic label shape: {label_data_dynamic.shape}")

# %% [markdown]
# ### Loading FGFR4 Proteomic data and Dynamic Features


# %%
# Load FGFR4 RNASeq dataset
loading_code = "goncalves-gdsc-1-FGFR_0939-LN_IC50-sin"
fgfr4_RNASeq_feature_data, fgfr4_RNASeq_label_data = data_link.get_data_using_code(
    loading_code
)

print(f"feature data shape: {fgfr4_RNASeq_feature_data.shape}")
print(f"label data shape: {fgfr4_RNASeq_label_data.shape}")

# %%

feature_data_dynamic, label_data_dynamic = data_link.get_data_using_code('generic-gdsc-1-FGFR_0939-LN_IC50-fgfr4_dynamic_features_v3_proteomic-true-Row')
print(f"Dynamic dataset shape: {feature_data_dynamic.shape}")
print(f"Dynamic label shape: {label_data_dynamic.shape}")

# %% [markdown]
# ## Functions


