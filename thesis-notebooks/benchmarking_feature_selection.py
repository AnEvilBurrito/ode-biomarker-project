# Jupyter notebook in Jupytext format

# %% [markdown]
# ## Initialisation

# %%
import os

# get current working directory
path = os.getcwd()

# find the string 'project' in the path, return index (fallback to cwd if not found)
index_project = path.find('project')
project_path = path[: index_project + 7] if index_project != -1 else path

# set the working directory to project root (as in the notebook)
os.chdir(project_path)
print(f'Project path set to: {os.getcwd()}')

from PathLoader import PathLoader  # noqa: E402
from DataLink import DataLink  # noqa: E402

# environment filenames used in the notebook
path_loader = PathLoader('data_config.env', 'current_user.env')
data_link = DataLink(path_loader, 'data_codes.csv')

# output folder and experiment id (same as notebook)
folder_name = "ThesisResult3-BenchmarkingFeatureSelection"
exp_id = "v1"

# prepare results directory and file save path
save_dir = os.path.join(path_loader.get_data_path(), 'data', 'results', folder_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
file_save_path = f"{save_dir}{os.sep}"

# Example dataset loading codes from the notebook (attempted but protected with try/except)
loading_code = "goncalves-gdsc-2-Palbociclib-LN_IC50-sin"
try:
    proteomic_feature_data, proteomic_label_data = data_link.get_data_using_code(loading_code)
    print(f'Proteomic feature data shape: {proteomic_feature_data.shape}', f'Proteomic label data shape: {proteomic_label_data.shape}')
except Exception as e:
    print(f'Could not load proteomic data for code {loading_code}: {e}')

loading_code = "ccle-gdsc-2-Palbociclib-LN_IC50"
try:
    ccle_feature_data, ccle_label_data = data_link.get_data_using_code(loading_code)
    print(f'CCLE feature data shape: {ccle_feature_data.shape}', f'CCLE label data shape: {ccle_label_data.shape}')
except Exception as e:
    print(f'Could not load CCLE data for code {loading_code}: {e}')

# %% [markdown]
# ## Functions 

# %% [markdown]
# ### Feature Selection Methods

# %%



# %% [markdown]
# ## Execution 


# %% [markdown]
# ## Results and Visualisation
