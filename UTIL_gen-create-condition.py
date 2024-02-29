import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from PathLoader import PathLoader
from DataLink import DataLink 

INPUT_DATA_CODE_EXPRESSION_DATA = 'ccle'
INPUT_DATA_CODE_MATCH_RULES = 'fgfr4_model_ccle_match_rules'
PARAM_FOLDER_NAME = 'fgfr4_model_initial_conditions'
PARAM_COMBINATION_METHOD = 'median' # weighted_median, median and cell_line_specific
SPECIFIC_CELL_LINE = 'MCF7' # only used when PARAM_COMBINATION_METHOD is 'cell_line_specific'
SILENT = False
SAVE_RESULT = True

### Bring in CCLE data
path_loader = PathLoader('data_config.env', 'current_user.env')
data_link = DataLink(path_loader, 'data_codes.csv')

if not SILENT: print('Loading data...')

### INPUT Data Code for m x n table of expression data (pkl pandas dataframe)
data_link.load_data_code(INPUT_DATA_CODE_EXPRESSION_DATA)
expression_df = data_link.data_code_database[INPUT_DATA_CODE_EXPRESSION_DATA]
if INPUT_DATA_CODE_EXPRESSION_DATA == 'ccle':
    expression_df.set_index('CELLLINE', inplace=True)
    
# INPUT Data code for match rules table (csv file)
data_link.load_data_code(INPUT_DATA_CODE_MATCH_RULES, verbose=True)
match_rules_df = data_link.data_code_database[INPUT_DATA_CODE_MATCH_RULES]

if not SILENT: print('Data loaded, processing...')

# iterate each row in expression_df

dataset_constructor = {}

if PARAM_COMBINATION_METHOD == 'median':
    # first compute the median expression values for each gene/protein
    median_expression_values = {}
    for col in expression_df.columns:
        expression_col = expression_df[col]
        expression_col_no_zero = expression_col[expression_col != 0] # ensure no zero values 
        median_expression_values[col] = expression_col_no_zero.median()

for i, row in tqdm(expression_df.iterrows(), total=expression_df.shape[0], disable=SILENT):
    model = row.name
    row_constructor = []
    
    for j, specie_info in match_rules_df.iterrows():
        specie = specie_info['specie']
        initial_value = specie_info['initial_value']
        references = specie_info['reference']
        # print(specie, initial_value, references)
        ### three transformation methods here
        if PARAM_COMBINATION_METHOD == 'weighted_median':
            raise NotImplementedError('weighted method not implemented yet')
        elif PARAM_COMBINATION_METHOD == 'median':
            
            if isinstance(references, float):
                # references is nan, in this case, use the default value for the specie
                row_constructor.append(initial_value)
            
            else:
                
                references = references.split(';')
                
                if len(references) > 1: 
                    # when there is more than one reference, take the average of the median values
                    # for all references
                    sum_vals = 0
                    for feature_name in references:
                        feature_value, feature_median_value = row[feature_name], median_expression_values[feature_name]
                        sum_vals += feature_value / feature_median_value
                        
                    norm_value = sum_vals / len(references) 
                    specie_value = norm_value * initial_value
                    row_constructor.append(specie_value)
                        
                else:
                    feature_name = references[0]
                    feature_value, feature_median_value, default_specie_value = row[feature_name], median_expression_values[feature_name], initial_value  
                    norm_value = feature_value / feature_median_value    
                    specie_value = norm_value * default_specie_value
                    row_constructor.append(specie_value)
                
            # print(j, specie, len(row_constructor))
                           
        elif PARAM_COMBINATION_METHOD == 'cell_line_specific':
            raise NotImplementedError('cell_line_specific method not implemented yet')
        else:
            raise ValueError('PARAM_COMBINATION_METHOD not recognized')
         
        # append row_constructor to dataset_constructor
    dataset_constructor[model] = row_constructor
    # break
    

columns = list(match_rules_df['specie'])
dynamic_features_df = pd.DataFrame.from_dict(dataset_constructor, orient='index', columns=columns)

if not SILENT: print(dynamic_features_df.head())
if not SILENT: print(dynamic_features_df.shape)

# --- creating folder name and path
if SAVE_RESULT:
    folder_name = PARAM_FOLDER_NAME

    if not os.path.exists(f'{path_loader.get_data_path()}data/results/{folder_name}'):
        os.makedirs(f'{path_loader.get_data_path()}data/results/{folder_name}')

    file_save_path = f'{path_loader.get_data_path()}data/results/{folder_name}/'

    dynamic_features_df.to_csv(f'{file_save_path}initial_conditions.csv')
    print('Done, saved to file')
