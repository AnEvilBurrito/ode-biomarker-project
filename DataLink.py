from PathLoader import PathLoader

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
            
    
    def get_data_using_code(loading_code: str):
        '''
        specialised processing method for convenient processing of raw data into typical feature & label data as 
        pandas dataframes. 
        
        Loading codes: 
            'ccle-gdsc-{number}-{drug_name}-{target_label}': combining CCLE and GDSC data to create a single dataset for a given drug
            'goncalves-gdsc-{number}-{drug_name}-{target_label}': combining goncalves and GDSC data to create a single dataset for a given drug
            'sy-cancercell2022': SY's processed data from Cancer Cell 2022            
        
        '''
        
        if 'ccle-gdsc' in loading_code: 
            # automated combination of CCLE and GDSC1/2 data can be loaded 
            pass 
        
        
        if 'goncalve-gdsc' in loading_code: 
            # automated combination of goncalves and GDSC1/2 data can be loaded 
            pass 