from PathLoader import PathLoader

import pandas as pd
import pickle 

class DataLink: 
    
    def __init__(self, path_loader: PathLoader, data_code_database_path) -> None:
        
        self.pathsHandle = path_loader 
        self.data_code_database_path = pd.read_csv(data_code_database_path)
        
        
        print('Loading data from biomarker data repository..')
        
        # import GDSC1 drug response data using pickle
        with open(f'{path_loader.get_data_path()}data/drug-response/GDSC1/cache_gdsc1.pkl', 'rb') as f:
            gdsc1 = pickle.load(f)
            gdsc1_info = pickle.load(f)
        
        self.gdsc1 = gdsc1
        self.gdsc1_info = gdsc1_info
        
        # import GDSC2 drug response data using pickle
        with open(f'{path_loader.get_data_path()}data/drug-response/GDSC2/cache_gdsc2.pkl', 'rb') as f:
            gdsc2 = pickle.load(f)
            gdsc2_info = pickle.load(f)
            
        self.gdsc2 = gdsc2
        self.gdsc2_info = gdsc2_info
            
        # import CCLE gene expression data using pickle

        with open(f'{path_loader.get_data_path()}data/gene-expression/CCLE_Public_22Q2/ccle_expression.pkl', 'rb') as f:
            gene_entrez = pickle.load(f)
            ccle = pickle.load(f)

        # import CCLE sample info data using pickle

        with open(f'{path_loader.get_data_path()}data/gene-expression/CCLE_Public_22Q2/ccle_sample_info.pkl', 'rb') as f:
            ccle_sample_info = pickle.load(f)
            
        self.gene_entrez = gene_entrez
        self.ccle = ccle
        self.ccle_sample_info = ccle_sample_info

        # import proteomic expression
        with open(f'{path_loader.get_data_path()}data/proteomic-expression/goncalves-2022-cell/goncalve_proteome_fillna_processed.pkl', 'rb') as f:
            joined_full_protein_matrix = pickle.load(f)
            joined_sin_peptile_exclusion_matrix = pickle.load(f)

        self.joined_full_protein_matrix = joined_full_protein_matrix
        self.joined_sin_peptile_exclusion_matrix = joined_sin_peptile_exclusion_matrix
        
    
    def load_data_code(self, data_code):
        
        pass 
    
    def load_data_code_raw(self, data_code, index_position, file_path):
        
        pass
    
    def get_data_using_code(loading_code: str):
        
        
        if 'ccle-gdsc' in loading_code: 
            # automated combination of CCLE and GDSC1/2 data can be loaded 
            pass 
        
        
        if 'goncalve-gdsc' in loading_code: 
            # automated combination of goncalves and GDSC1/2 data can be loaded 
            pass 
        
        