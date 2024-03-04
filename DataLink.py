from PathLoader import PathLoader
import DataFunctions 

import pandas as pd
import pickle 

class DataLink: 
    
    def __init__(self, path_loader: PathLoader, data_code_database_path) -> None:
        
        self.paths_handle = path_loader 
        self.data_code_database_path = pd.read_csv(data_code_database_path)
        self.data_code_database = {}
        
    def get_data_from_code(self, data_code, automatic_load=True, verbose=False):
        if data_code in self.data_code_database.keys():
            return self.data_code_database[data_code]
        else:
            # attempt automatic load 
            if automatic_load:
                self.load_data_code(data_code, verbose=verbose)
                return self.data_code_database[data_code]
            else:
                raise Exception(f'Data code {data_code} not found in database. Use load_data_code() to load data.')

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
            
    def load_data_code_raw(self, data_code, index_position, file_path, verbose=False, enforce_raw=True):
        
        # handles csv, xlsx and pickle files
        '''
        file_path: string, path to the file to load
        index_position: int, index position of the data to load from a pickle file
        enforce_raw: bool, if True, the raw file will be loaded, if False, a cached pkl file will be loaded if available
        '''
        
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(f'{self.paths_handle.get_data_path()}{file_path}')
            self.data_code_database[data_code] = df
            if verbose:
                print(f'Data code {data_code} loaded at {file_path} with index position {index_position}.')
        
        if file_path.endswith('.csv'):
            
            # attempt to load a cached pkl file first
            if not enforce_raw:
                pkl_file_path = file_path.replace('.csv', '.pkl')
                try:
                    with open(f'{self.paths_handle.get_data_path()}{pkl_file_path}', 'rb') as f:
                        for i in range(index_position+1):
                            data = pickle.load(f)
                            if i == index_position:
                                self.data_code_database[data_code] = data
                                if verbose:
                                    print(
                                        f'Pickle Cached Version of Data code {data_code} loaded at {file_path} with index position {index_position}. Enforced raw loading: {enforce_raw}')
                                return
                except FileNotFoundError:
                    pass
            
            df = pd.read_csv(f'{self.paths_handle.get_data_path()}{file_path}')
            csv_file_loaded = True
            self.data_code_database[data_code] = df
            if verbose:
                print(f'Data code {data_code} loaded at {file_path} with index position {index_position}. Enforced raw loading: {enforce_raw}')
            
            if not enforce_raw:       
                if verbose:
                    print(f'Creating a cached pkl file at {pkl_file_path}...')
                
                with open(f'{self.paths_handle.get_data_path()}{pkl_file_path}', 'wb') as f:
                    pickle.dump(df, f)
        
        elif file_path.endswith('.pkl'):

            found = False
            with open(f'{self.paths_handle.get_data_path()}{file_path}', 'rb') as f:
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
            'ccle-gdsc-{number}-{drug_name}-{target_label}': combining CCLE and GDSC data to create a single dataset for a given drug \n
            'goncalves-gdsc-{number}-{drug_name}-{target_label}-{full/sin}': combining goncalves and GDSC data to create a single dataset for a given drug \n
            'sy-cancercell2022': SY's processed data from Cancer Cell 2022 \n       
            'anthony-ode-gdsc-{number}-{drug_name}-{target_label}-{norm/default}': combining anthony's dynamic data and GDSC data to create a single dataset for a given drug \n
            'generic-gdsc-{number}-{drug_name}-{target_label}-{dataset_name}-{replace_index}-{row_index}': 
                combining generic data and GDSC data to create a single dataset for a given drug, 
                dataset rows must be in Sanger Model ID format. `dataset_name` will be the loading
                code for any features of a set of cell lines\n
            
        Returns always two paramters in the following order: 
            feature_data: pandas dataframe of feature data
            label_data: pandas dataframe of label data    
        '''
        
        if 'generic-gdsc' in loading_code:
            # automated combination of generic data with sanger model IDs as index and GDSC1/2 data can be loaded 
            splitted_code = loading_code.split('-')
            # replace_index and row_index are optional parameters
            # replace_index: if true, it is assumed the dataset contains DepMap_Ids, 
            #                the index of the generic dataset wil be replaced with Sanger Model IDs
            # row_index: if replace_index is true, this parameter will be used to specify the 
            #            column name of the dataset that contains the DepMap_IDs
            gdsc_num, drug_name, target_label, dataset_name, replace_index, row_index = splitted_code[2], splitted_code[3], splitted_code[4], splitted_code[5], splitted_code[6], splitted_code[7]
            
            if f'gdsc{gdsc_num}' not in self.data_code_database.keys():
                self.load_data_code(f'gdsc{gdsc_num}')
            gdsc = self.data_code_database[f'gdsc{gdsc_num}']
            if dataset_name not in self.data_code_database.keys():
                self.load_data_code(dataset_name)
            generic_data = self.data_code_database[dataset_name]
            
            # handling optional parameters
            if replace_index == 'true':
                # run the index replacement function from DataFunctions
                if 'ccle_sample_info' not in self.data_code_database.keys():
                    self.load_data_code('ccle_sample_info')
                ccle_sample_info = self.data_code_database['ccle_sample_info']
                
                if row_index != 'index':
                    # this option is used for datasets that have DepMap_IDs in a column other than the index
                    processed_data = generic_data.set_index(row_index)
                processed_data = DataFunctions.dataset_to_sanger_model_id_from_ccle(processed_data, ccle_sample_info)
            else: 
                processed_data = generic_data
                
            whole_df = DataFunctions.create_joint_dataset_from_proteome_gdsc(drug_name, processed_data, gdsc, target_label)
            feature_data, label_data = DataFunctions.create_feature_and_label(whole_df, label_name=target_label)
            
            return feature_data, label_data
        
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
        
        if 'goncalves-gdsc' in loading_code: 
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
        
        if 'anthony-ode-gdsc' in loading_code: 
            
            splitted_code = loading_code.split('-')
            gdsc_num, drug_name, target_label, norm_or_default = splitted_code[3], splitted_code[4], splitted_code[5], splitted_code[6]
            
            if norm_or_default == 'norm':
                data_code_to_load = 'dynamic_features_norm'
            elif norm_or_default == 'default':
                data_code_to_load = 'dynamic_features'
            else:
                raise Exception(f'Invalid data code for loading anthony-dynamic-gdsc at the end: {norm_or_default}. only "norm" or "default" is allowed.')
            
            if data_code_to_load not in self.data_code_database.keys():
                self.load_data_code(data_code_to_load)
            dynamic_data = self.data_code_database[data_code_to_load]
            
            if f'gdsc{gdsc_num}' not in self.data_code_database.keys():
                self.load_data_code(f'gdsc{gdsc_num}')
            gdsc = self.data_code_database[f'gdsc{gdsc_num}']
            
            ccle_sample_info = self.get_data_from_code('ccle_sample_info')
            
            depmap_to_sanger = ccle_sample_info[['DepMap_ID', 'Sanger_Model_ID']]
            depmap_to_sanger = depmap_to_sanger.dropna(subset=['Sanger_Model_ID'])
            
            dynamic_features = dynamic_data.join(depmap_to_sanger.set_index('DepMap_ID'), on='Unnamed: 0')
            dynamic_features.drop(columns=['Unnamed: 0'], inplace=True)
            dynamic_features.set_index('Sanger_Model_ID', inplace=True)
        
            whole_df = DataFunctions.create_joint_dataset_from_proteome_gdsc(drug_name, dynamic_features, gdsc, target_label)
            feature_data, label_data = DataFunctions.create_feature_and_label(whole_df, label_name=target_label)
            
            return feature_data, label_data