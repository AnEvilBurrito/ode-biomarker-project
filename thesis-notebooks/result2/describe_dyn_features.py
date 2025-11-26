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


# %%
# Load Transcriptomics Palbociclib dataset
loading_code = "ccle-gdsc-2-Palbociclib-LN_IC50"
rnaseq_feature_data, rnaseq_label_data = data_link.get_data_using_code(
    loading_code
)

print(f"RNASeq feature data shape: {rnaseq_feature_data.shape}")
print(f"RNASeq label data shape: {rnaseq_label_data.shape}")

# %%

feature_data_dynamic, label_data_dynamic = data_link.get_data_using_code('generic-gdsc-2-Palbociclib-LN_IC50-cdk46_ccle_dynamic_features_v4_ccle-true-Unnamed: 0')
print(f"Dynamic dataset shape: {feature_data_dynamic.shape}")
print(f"Dynamic label shape: {label_data_dynamic.shape}")

# %% [markdown]
# ## Functions


