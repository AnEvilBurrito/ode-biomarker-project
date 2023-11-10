# This script is used to generate individualised specie initial conditions based on CCLE data and model default initial conditions 

import os

import numpy as np
import pandas as pd

if __name__ == "__main__": 
    
    ### Bring in CCLE data
    from PathLoader import PathLoader
    from DataLink import DataLink 
    path_loader = PathLoader('data_config.env', 'current_user.env')
    TheLink = DataLink(path_loader, 'data_codes.csv')
    TheLink.load_data_code('ccle')
    ccle_df = TheLink.data_code_database['ccle']
    

    ### Bring in model default initial conditions from sbml file
    import roadrunner
    rr = roadrunner.RoadRunner("data\export_ECC_Base.xml")
    species = rr.model.getFloatingSpeciesIds()
    for idx, specie in enumerate(species):
        print(f'{idx} {specie} init {rr.model[f"init({specie})"]} curr {rr.model[specie]}')
    
    ### Bring in spreadsheet for matching rules between CCLE and model species
    
    match_rules_file = TheLink.get_data_from_code('integrate_ccle_anthony')
    match_rules_files_dropna = match_rules_file.dropna(subset=['CCLE reference'])
    
    species_ccle_matches = {}
    for i in range(len(match_rules_files_dropna)):
        row = match_rules_files_dropna.iloc[i]  
        specie_name = row['Protein Name']
        ccle_matches = row['CCLE reference']
        ccle_matches = ccle_matches.split(';')
        print(f'{specie_name}: {ccle_matches}')
        species_ccle_matches[specie_name] = ccle_matches
    
    ### Bring in the initial conditions for the species from best parameter sets for consistency
    
    best_paramsets = TheLink.get_data_from_code('best_paramsets_anthony')
    params_row = best_paramsets.iloc[0]
    
    species_value_dict = {}
    for col in params_row.index:
        if col in species:
            model_specie_value = params_row[col]
            # print(f'species {col} set to {params_row[col]}')
            species_value_dict[col] = model_specie_value
            
            
    ### Begin to create the initial conditions for the species 
    
    dataset = []
    combination_method = 'weighted'
    
    for specie_name, specie_value in species_value_dict.items():
        if specie_name in species_ccle_matches:
            matches = species_ccle_matches[specie_name]
            if len(matches) > 1: 
                print(f'COMBINATION {specie_name} {specie_value} {species_ccle_matches[specie_name]}')
                # combination normalisation method 
                # two options:
                #   1. average combination 
                #   2. weighted by sample size combination
                if combination_method == 'average': 
                    N = len(matches)
                    all_columns = []
                    # debug operations 
                    medians = []
                    sample_row_vals = []
                    sample_ccle_vals = []
                    for match in matches: 
                        gene_column = ccle_df[match]
                        gene_column_no_zero = gene_column[gene_column != 0]
                        m = gene_column_no_zero.median()
                        s = specie_value
                        normalised_column = gene_column / m
                        all_columns.append(normalised_column)
                        # following are debug operations
                        medians.append(m)
                        sample_row_vals.append(normalised_column[0])
                        sample_ccle_vals.append(gene_column[0])
                        
                    all_columns = pd.concat(all_columns, axis=1)
                    # sum row wise and divide by N, multiply by specie value
                    sum_row_columns = all_columns.sum(axis=1) / N * s
                    
                    # verify logic 
                    # get the first value of each column in all_columns

                    # print('medians', medians)
                    # print('ccle vals', sample_ccle_vals)
                    # print('transformed vals', sample_row_vals)
                    # print('final ratio',all_columns.sum(axis=1)[0] / N)
                    # print('multiply by initial cond', sum_row_columns[0], sum_row_columns.shape)
                    
                    # final append
                    sum_row_columns = list(sum_row_columns)
                    dataset.append(sum_row_columns) 
                    
                elif combination_method == 'weighted': 
                    # TODO: once this is done, this script is largely complete
                    all_columns = []
                    length_vector = []
                    for match in matches:
                        gene_column = ccle_df[match]
                        gene_column_no_zero = gene_column[gene_column != 0]
                        length = len(gene_column_no_zero)
                        length_vector.append(length)
                        m = gene_column_no_zero.median()
                        s = specie_value
                        normalised_column = gene_column / m
                        all_columns.append(normalised_column)
                    
                    all_columns = pd.concat(all_columns, axis=1)
                    length_vector = np.array(length_vector)
                    length_vector = length_vector / length_vector.sum()
                    
                    # multiply by length vector
                    transformed_columns = all_columns * length_vector
                    # sum row wise and multiply by specie value
                    sum_row_columns = transformed_columns.sum(axis=1) * s
                    
                    # verify logic
                    # print('length vector', length_vector)
                    # print('all columns', list(all_columns.iloc[0]))
                    # print('transformed vals', list(transformed_columns.iloc[0]))
                    # print('final ratio', transformed_columns.sum(axis=1)[0])
                    # print('sum row columns', sum_row_columns[0], sum_row_columns.shape)
                    
                    
                    # final append
                    sum_row_columns = list(sum_row_columns)
                    dataset.append(sum_row_columns)
                    
            elif len(matches) == 1:
                # direct normalisation method 
                print(f'DIRECT {specie_name} {specie_value} {species_ccle_matches[specie_name]}')
                gene_column = ccle_df[species_ccle_matches[specie_name][0]]
                gene_column_no_zero = gene_column[gene_column != 0]
                m = gene_column_no_zero.median()
                s = specie_value
                species_column = gene_column / m * s 
                species_column = list(species_column)
                print(species_column[0])
                dataset.append(species_column) 
                
            else: 
                # throw error
                raise ValueError(f'No matches for {specie_name}')
        else: 
            # replace with default value 
            print(f'REPLACE {specie_name} {specie_value}')
            
            specie_value_column = [specie_value] * ccle_df.shape[0]
            dataset.append(specie_value_column)
    
    new_df = pd.DataFrame(dataset).transpose()
    new_df.columns = species
    new_df.index = ccle_df['CELLLINE']
    print(new_df.head())
    print(new_df.shape)
    
    # --- creating folder name and path
    folder_name = 'create-initial-conditions'

    if not os.path.exists(f'{path_loader.get_data_path()}data/results/{folder_name}'):
        os.makedirs(f'{path_loader.get_data_path()}data/results/{folder_name}')

    file_save_path = f'{path_loader.get_data_path()}data/results/{folder_name}/'

    new_df.to_csv(f'{file_save_path}initial_conditions.csv')
    print('Done, saved to file')