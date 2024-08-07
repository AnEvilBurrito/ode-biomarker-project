import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from PathLoader import PathLoader
from DataLink import DataLink

INPUT_DATA_CODE_EXPRESSION_DATA = 'ccle'
EXPRESSION_DATA_INDEX_COL = 'CELLLINE'  # 'CELLLINE' for 'ccle', 'Cell_Line' for 'ccle_protein_expression'
INPUT_DATA_CODE_MATCH_RULES = 'cdk_model_match_rules'
PARAM_OUTPUT_FOLDER_NAME = 'create-initial-conditions'
PARAM_COMBINATION_METHOD = 'median' # weighted_median, median and cell_line_specific
SPECIFIC_CELL_LINE = 'MCF7' # only used when PARAM_COMBINATION_METHOD is 'cell_line_specific'
SILENT = False
SAVE_RESULT = False
DISPLAY_RESULT = True

# display the parameters
if not SILENT:
    print('PARAM_COMBINATION_METHOD: ', PARAM_COMBINATION_METHOD)
    print('SPECIFIC_CELL_LINE: ', SPECIFIC_CELL_LINE)
    print('SILENT: ', SILENT)
    print('SAVE_RESULT: ', SAVE_RESULT)
    print('DISPLAY_RESULT: ', DISPLAY_RESULT)
    print('INPUT_DATA_CODE_EXPRESSION_DATA: ', INPUT_DATA_CODE_EXPRESSION_DATA)
    print('INPUT_DATA_CODE_MATCH_RULES: ', INPUT_DATA_CODE_MATCH_RULES)
    print('PARAM_FOLDER_NAME: ', PARAM_OUTPUT_FOLDER_NAME)
    print('------------------')

### Paths Configurations
path_loader = PathLoader('data_config.env', 'current_user.env')
data_link = DataLink(path_loader, 'data_codes.csv')

if not SILENT: print('Loading data codes...')

### INPUT Data Code for m x n table of expression data (pkl pandas dataframe)
data_link.load_data_code(INPUT_DATA_CODE_EXPRESSION_DATA)
expression_df = data_link.data_code_database[INPUT_DATA_CODE_EXPRESSION_DATA]
expression_df.set_index(EXPRESSION_DATA_INDEX_COL, inplace=True)
    
# INPUT Data code for match rules table (csv file)
data_link.load_data_code(INPUT_DATA_CODE_MATCH_RULES, verbose=True)
match_rules_df = data_link.data_code_database[INPUT_DATA_CODE_MATCH_RULES]

if not SILENT: print('Data loaded successfully, processing data now...')

# iterate each row in expression_df
dataset_constructor = {}


ambiguous_median_count = 0
negative_specie_value_count = 0
nan_specie_value_count = 0

if PARAM_COMBINATION_METHOD == 'median':
    # first compute the median expression values for each gene/protein
    median_expression_values = {}
    for col in expression_df.columns:
        expression_col = expression_df[col]
        expression_col_no_zero = expression_col[expression_col != 0] # ensure no zero values 
        median_values = expression_col_no_zero.median()
        # in the case of multiple median values caused by protein isoforms, take the average
        if isinstance(median_values, pd.Series):
            median_values = median_values.dropna()
            median_values = median_values.mean()
            ambiguous_median_count += 1
            # print(f'Warning: multiple median values detected for {col}, taking the average: {median_values} instead of {expression_col_no_zero.median()}')
        median_expression_values[col] = median_values
        # print(f'col: {col}, median: {median_expression_values[col]}')
        

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
                        if isinstance(feature_value, pd.Series):
                            feature_value = feature_value.mean()
                        # print(f'feature_name: {feature_name}, feature_value: {feature_value}, feature_median_value: {feature_median_value}')
                        sum_vals += feature_value / feature_median_value
                        
                    norm_value = sum_vals / len(references) 
                    specie_value = norm_value * initial_value
                    # print('Multiple', feature_name, type(specie_value), specie_value, norm_value, initial_value)
                    if specie_value < 0:
                        # print('Warning: negative value detected, setting to default value')
                        specie_value = initial_value
                        negative_specie_value_count += 1

                    if np.isnan(specie_value):
                        specie_value = initial_value
                        nan_specie_value_count += 1
                    row_constructor.append(specie_value)
                        
                else:
                    feature_name = references[0]
                    feature_value, feature_median_value, default_specie_value = row[feature_name], median_expression_values[feature_name], initial_value  
                    if isinstance(feature_value, pd.Series):
                        feature_value = feature_value.mean()
                    norm_value = feature_value / feature_median_value    
                    specie_value = norm_value * default_specie_value
                    # print('Single', feature_name, type(specie_value), specie_value)
                    if specie_value < 0:
                        # print('Warning: negative value detected, setting to default value')
                        specie_value = initial_value
                        negative_specie_value_count += 1
                    
                    if np.isnan(specie_value):
                        # print('Warning: nan value detected, setting to default value')
                        specie_value = initial_value  
                        nan_specie_value_count += 1                  
                    row_constructor.append(specie_value)
                
                    
                
            # print(j, specie, len(row_constructor))
                           
        elif PARAM_COMBINATION_METHOD == 'cell_line_specific':
            raise NotImplementedError('cell_line_specific method not implemented yet')
        else:
            raise ValueError('PARAM_COMBINATION_METHOD not recognized')
         
        # append row_constructor to dataset_constructor
    dataset_constructor[model] = row_constructor
    
columns = list(match_rules_df['specie'])
dynamic_features_df = pd.DataFrame.from_dict(dataset_constructor, orient='index', columns=columns)

if SAVE_RESULT: 
    folder_name = PARAM_OUTPUT_FOLDER_NAME

    if not os.path.exists(f'{path_loader.get_data_path()}data/results/{folder_name}'):
        os.makedirs(f'{path_loader.get_data_path()}data/results/{folder_name}')

    file_save_path = f'{path_loader.get_data_path()}data/results/{folder_name}/'
    
    param_str = f'{PARAM_COMBINATION_METHOD}-{INPUT_DATA_CODE_EXPRESSION_DATA}-{INPUT_DATA_CODE_MATCH_RULES}-{dynamic_features_df.shape[0]}x{dynamic_features_df.shape[1]}-initial_conditions.csv'
    
    dynamic_features_df.to_csv(f'{file_save_path}{param_str}.csv')
    print(f'Done, saved to file as: {param_str}.csv')
    
if DISPLAY_RESULT:
    print('Ambiguous median values detected: ', ambiguous_median_count)
    print('Negative specie values detected: ', negative_specie_value_count)
    print('Nan specie values detected: ', nan_specie_value_count)
    print('Shape of original data: ', expression_df.shape)
    print('dataframe shape: ', dynamic_features_df.shape, 'printing the first 5 rows of the dataframe:')
    print(dynamic_features_df.head())