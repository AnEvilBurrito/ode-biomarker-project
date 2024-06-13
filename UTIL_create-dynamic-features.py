# This script is used to generate individualised specie initial conditions based on CCLE data and model default initial conditions 

import os 

import numpy as np
import pandas as pd
from tqdm import tqdm

# HEADER PARAMETERS
PARAM_INPUT_DATA_CODE = 'cdk_model_downsampled_simulation_v2_ccle_proteomics'
PARAM_FOLDER_NAME = 'dynamic-features-CDK-AH-v4-proteomics'
PARAM_NORMALISE_TIME_BASED_VALUES = True 

# function to calculate dynamic simulation features for a specie and cell line

if __name__ == "__main__": 
    
    ### Load in dynamic features file 
    from PathLoader import PathLoader
    from DataLink import DataLink
    from get_dynamic_features import get_dynamic_features
    path_loader = PathLoader('data_config.env', 'current_user.env')
    TheLink = DataLink(path_loader, 'data_codes.csv')
    
    print('Loading data..')
    simulation_data = TheLink.get_data_from_code(PARAM_INPUT_DATA_CODE)

    all_species = list(simulation_data.columns[2:])
    all_celllines = simulation_data['Cellline'].unique()
    if 'Time' in all_species:
        all_species.remove('Time')

    print('Calculating dynamic features..')
    new_dataset = []
    for c in tqdm(all_celllines):
        cellline_dynamic_features = []
        for s in all_species:
            selected_data = simulation_data[simulation_data['Cellline'] == c]
            specie_data = selected_data[s]
            dyn_feats = get_dynamic_features(specie_data, normalise=PARAM_NORMALISE_TIME_BASED_VALUES)
            cellline_dynamic_features.extend(dyn_feats)
        new_dataset.append(cellline_dynamic_features)
        

    dynamic_feature_label = ['auc', 'median', 'tfc', 'tmax', 'max', 'tmin', 'min', 'ttsv', 'tsv', 'init']    
    new_df = pd.DataFrame(new_dataset, columns=[s + '_' + dynamic_feature for s in all_species for dynamic_feature in dynamic_feature_label], index=all_celllines)
    
    # --- creating folder name and path
    folder_name = PARAM_FOLDER_NAME

    if not os.path.exists(f'{path_loader.get_data_path()}data/results/{folder_name}'):
        os.makedirs(f'{path_loader.get_data_path()}data/results/{folder_name}')

    file_save_path = f'{path_loader.get_data_path()}data/results/{folder_name}/'

    file_name = f'ode_dynamic_features_time_norm.csv' if PARAM_NORMALISE_TIME_BASED_VALUES else f'ode_dynamic_features.csv'
    
    new_df.to_csv(f'{file_save_path}{file_name}')
    print('Done, saved to file')
