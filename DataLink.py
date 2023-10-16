from PathLoader import PathLoader
import DataFunctions 

import pandas as pd
import pickle 

class DataLink: 
    
    def __init__(self, path_loader: PathLoader, data_code_database_path) -> None:
        
        self.pathsHandle = path_loader 
        self.data_code_database_path = pd.read_csv(data_code_database_path)
        self.data_code_database = {}

    def load_all(self, verbose=True):
        
        for data_code in self.data_code_database_path['data_code']:
            self.load_data_code(data_code, verbose=verbose)
    
    def load_data_code(self, data_code, verbose=False):
        
        if data_code in self.data_code_database_path['data_code'].values:
            index_pos = self.data_code_database_path[self.data_code_database_path['data_code'] == data_code]['index_position'].values[0]
            file_path = self.data_code_database_path[self.data_code_database_path['data_code'] == data_code]['file_path'].values[0]
            self.load_data_code_raw(data_code, index_pos, file_path, verbose=verbose)
        
        else: 
            raise Exception(f'Data code {data_code} not found in database.')
            
    def load_data_code_raw(self, data_code, index_position, file_path, verbose=False):
        
        # handles both csv and pickle files
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(f'{self.pathsHandle.get_data_path()}{file_path}')
            self.data_code_database[data_code] = df
            if verbose:
                print(f'Data code {data_code} loaded at {file_path} with index position {index_position}.')
        
        elif file_path.endswith('.pkl'):

            found = False
            with open(f'{self.pathsHandle.get_data_path()}{file_path}', 'rb') as f:
                for i in range(index_position+1):
                    data = pickle.load(f)
                    if i == index_position:
                        self.data_code_database[data_code] = data
                        found = True
                        if verbose:
                            print(f'Data code {data_code} loaded at {file_path} with index position {index_position}.')
                        break
                    
            if not found:
                raise Exception(f'Data code {data_code} not found at {file_path} with index position {index_position}.')
            
    
    def get_data_using_code(self, loading_code: str):
        '''
        specialised processing method for convenient processing of raw data into typical feature & label data as 
        pandas dataframes. 
        
        Loading codes: 
            'ccle-gdsc-{number}-{drug_name}-{target_label}': combining CCLE and GDSC data to create a single dataset for a given drug
            'goncalves-gdsc-{number}-{drug_name}-{target_label}-{full}': combining goncalves and GDSC data to create a single dataset for a given drug
            'sy-cancercell2022': SY's processed data from Cancer Cell 2022        
            
        Returns always two paramters in the following order: 
            feature_data: pandas dataframe of feature data
            label_data: pandas dataframe of label data    
        '''
        
        if 'ccle-gdsc' in loading_code: 
            # automated combination of CCLE and GDSC1/2 data can be loaded 
            
            splitted_code = loading_code.split('-')
            gdsc_num, drug_name, target_label = splitted_code[2], splitted_code[3], splitted_code[4]
            # loading data from ccle 
            if 'ccle' not in self.data_code_database.keys():
                self.load_data_code('ccle')
            ccle = self.data_code_database['ccle']
            if 'ccle_sample_info' not in self.data_code_database.keys():
                self.load_data_code('ccle_sample_info')
            ccle_sample_info = self.data_code_database['ccle_sample_info']
            
            if f'gdsc{gdsc_num}' not in self.data_code_database.keys():
                self.load_data_code(f'gdsc{gdsc_num}')
            gdsc = self.data_code_database[f'gdsc{gdsc_num}']
            whole_df = DataFunctions.create_joint_dataset_from_ccle_gdsc2(drug_name, gdsc, ccle, ccle_sample_info, target_label)
            feature_data, label_data = DataFunctions.create_feature_and_label(whole_df, label_name=target_label)
            
            return feature_data, label_data
        
        if 'goncalve-gdsc' in loading_code: 
            # automated combination of goncalves and GDSC1/2 data can be loaded 
            splitted_code = loading_code.split('-')
            gdsc_num, drug_name, target_label, sin_or_full = splitted_code[2], splitted_code[3], splitted_code[4], splitted_code[5]
            
            if sin_or_full == 'full':
                data_code_to_load = 'full_protein_matrix'
            elif sin_or_full == 'sin':
                data_code_to_load = 'single_peptide_exclusion_matrix'
            else:
                raise Exception(f'Invalid data code for loading goncalve-gdsc at the end: {sin_or_full}. only "full" or "sin" is allowed.')
            
            if data_code_to_load not in self.data_code_database.keys():
                self.load_data_code(data_code_to_load)
            protein_matrix = self.data_code_database[data_code_to_load]
            
            if f'gdsc{gdsc_num}' not in self.data_code_database.keys():
                self.load_data_code(f'gdsc{gdsc_num}')
            gdsc = self.data_code_database[f'gdsc{gdsc_num}']
            
            whole_df = DataFunctions.create_joint_dataset_from_proteome_gdsc(drug_name, protein_matrix, gdsc, target_label)
            feature_data, label_data = DataFunctions.create_feature_and_label(whole_df, label_name=target_label)
            
            return feature_data, label_data
        
        
        if 'sy-cancercell2022' in loading_code:
            
            if 'sy_cancercell2022' not in self.data_code_database.keys():
                self.load_data_code('sy_cancercell2022')
            
            whole_df = self.data_code_database['sy_cancercell2022']
            whole_df_dropnan = whole_df.dropna(subset=['AUC']) # remove nan values in the label column
            whole_df_dropnan.set_index('Row', inplace=True)
            feature_data, label_data = DataFunctions.create_feature_and_label(whole_df_dropnan, label_name='AUC')
            
            return feature_data, label_data