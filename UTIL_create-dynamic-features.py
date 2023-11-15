# This script is used to generate individualised specie initial conditions based on CCLE data and model default initial conditions 

import os 

import numpy as np
import pandas as pd

# HEADER PARAMETERS
PARAM_INPUT_DATA_CODE = 'dynamic_simulation_data'
PARAM_FOLDER_NAME = 'create-dynamic-features'
PARAM_NORMALISE_TIME_BASED_VALUES = True 

# function to calculate dynamic simulation features for a specie and cell line

def calculate_dynamic_simulation_features(specie, cellline, dynamic_data, normalise_time_based_values=True):
    selected_data = dynamic_data[dynamic_data['Cellline'] == cellline]
    specie_data = selected_data[specie]
    # reset index to start from 0
    specie_data = specie_data.reset_index(drop=True)

    # calculate AUC
    auc = np.trapz(specie_data, dx=1)

    # obtain the max value of the specie
    max_value = specie_data.max()

    # obtain the time at which the max value occurs
    max_time = specie_data.idxmax()

    # obtain the min value of the specie
    min_value = specie_data.min()

    # obtain the time at which the min value occurs
    min_time = specie_data.idxmin()

    # mean value of the specie
    mean_value = specie_data.mean()

    # median value of the specie
    median_value = specie_data.median()

    # total fold change (TFC) from 0 to end
    start = specie_data.iloc[0]
    end = specie_data.iloc[-1]
    tfc = (end - start) / start

    # time to stable value (TSV), a time point t where the value of the specie no longer changes more than 0.01 for all t' > t
    tsv = specie_data.shape[0]
    change_abs_tolerance = 0.01
    difference = specie_data.diff()
    while tsv > 0:
        if abs(difference.iloc[tsv-1]) < change_abs_tolerance:
            tsv = tsv - 1
        else:
            break

    # normalise all time based values and AUC to the maximum simulation time
    max_sim_time = specie_data.shape[0]
    n_auc = auc / max_sim_time
    n_max_time = max_time / max_sim_time
    n_min_time = min_time / max_sim_time
    n_tsv = tsv / max_sim_time

    if normalise_time_based_values:
        return [n_auc, max_value, n_max_time, min_value, n_min_time, mean_value, median_value, tfc, n_tsv]

    return [auc, max_value, max_time, min_value, min_time, mean_value, median_value, tfc, tsv]

if __name__ == "__main__": 
    
    ### Load in dynamic features file 
    from PathLoader import PathLoader
    from DataLink import DataLink
    path_loader = PathLoader('data_config.env', 'current_user.env')
    TheLink = DataLink(path_loader, 'data_codes.csv')
    dynamic_data = TheLink.get_data_from_code(PARAM_INPUT_DATA_CODE)

    all_species = dynamic_data.columns[2:]
    all_celllines = dynamic_data['Cellline'].unique()

    new_dataset = []

    for c in all_celllines:
        cellline_dynamic_features = []
        for s in all_species:
            cellline_dynamic_features.extend(calculate_dynamic_simulation_features(s, c, dynamic_data, PARAM_NORMALISE_TIME_BASED_VALUES))
        new_dataset.append(cellline_dynamic_features)
        

    dynamic_feature_label = ['auc', 'max', 'max_time', 'min', 'min_time', 'mean', 'median', 'tfc', 'tsv']    
    new_df = pd.DataFrame(new_dataset, columns=[s + '_' + dynamic_feature for s in all_species for dynamic_feature in dynamic_feature_label], index=all_celllines)
    
    # --- creating folder name and path
    folder_name = PARAM_FOLDER_NAME

    if not os.path.exists(f'{path_loader.get_data_path()}data/results/{folder_name}'):
        os.makedirs(f'{path_loader.get_data_path()}data/results/{folder_name}')

    file_save_path = f'{path_loader.get_data_path()}data/results/{folder_name}/'

    new_df.to_csv(f'{file_save_path}anthony_model_dynamic_features.csv')
    print('Done, saved to file')
