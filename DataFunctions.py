import pandas as pd 
import numpy as np

def get_protein_id_by_name(name: str, info: pd.DataFrame, alias: pd.DataFrame, 
                           absolute_match = True,
                           edit_distance=1):
    # if name exist in the info dataframe, return the id
    # get the `#string_protein_id` column from the info dataframe using the `preferred_name` column, the 
    # param `name` is the value of the `preferred_name` column 

    # get the `#string_protein_id` column using name 
    name_id = info.loc[info['preferred_name'].str.lower() == name.lower()]['#string_protein_id']
    
    # if the name_id is not empty and only one value, return the value
    if not name_id.empty and len(name_id) == 1:
        return name_id.values[0]
    
    if len(name_id) > 1:
        print('Warning: more than one id found for the given name')
        return None
    
    if name_id.empty:  
        
        # get the `#string_protein_id` column from the alias dataframe using the `alias` column, the
        # param `name` is the value of the `alias` column
        alias_id = alias.loc[alias['alias'].str.lower() == name.lower()]['#string_protein_id']

        if len(alias_id) > 1:
            if alias_id.eq(alias_id.iloc[0]).all():
                return alias_id.values[0]
            else:
                print('Warning: more than one id found for the given name (alias)')
                print(alias_id)
                return None

        if not alias_id.empty and len(alias_id) == 1:
            return alias_id.values[0]
        
        if alias_id.empty:
            return None 



def get_protein_interactors(id: str, string_df: pd.DataFrame, score_threshold=900):
    # get the interactors of the protein with the given id
    # return a list of protein ids
    interactors = string_df.loc[string_df['protein1'] == id]['protein2']

    # get the interactors with the given score threshold
    interactors = interactors[string_df['combined_score'] > score_threshold]
    return interactors.values.tolist()

def get_protein_name_by_id(id: str, relation_df: pd.DataFrame, field_name: str, check_field_name: str = '#string_protein_id'):
    
    name = relation_df.loc[relation_df[check_field_name] == id][field_name]
    if not name.empty:
        return name.values[0]
    else:
        # print('Warning: no name found for the given id')
        return None


def run_test_get_protein_id_by_name():

    # load the dataframes with pickle 
    import pickle
    with open('data\protein-interaction\STRING\string_df.pkl', 'rb') as f:
        string_df = pickle.load(f)
        string_df_info = pickle.load(f)
        string_df_alias = pickle.load(f)


    with open('data/proteomic-expression/goncalves-2022-cell/goncalve_proteome_fillna.pkl', 'rb') as f:
        full_protein_matrix = pickle.load(f)
        sin_peptile_exclusion_matrix = pickle.load(f)
        goncalve_cell_line_info = pickle.load(f)


    # import CCLE gene expression data using pickle

    with open('data/gene-expression/CCLE_Public_22Q2/ccle_expression.pkl', 'rb') as f:
        gene_entrez = pickle.load(f)
        ccle = pickle.load(f)

    # import CCLE sample info data using pickle

    with open('data/gene-expression/CCLE_Public_22Q2/ccle_sample_info.pkl', 'rb') as f:
        ccle_sample_info = pickle.load(f)

    columns_protein = sin_peptile_exclusion_matrix.columns

    # perform the test
    test_id = get_protein_id_by_name('HSP90AA1', string_df_info, string_df_alias)
    print(test_id)

    test_id = get_protein_id_by_name('HSP90Aa1', string_df_info, string_df_alias, absolute_match=False)
    print(test_id)

    test_id = get_protein_id_by_name('HSP90A1', string_df_info, string_df_alias, absolute_match=False)
    print(test_id)

    for name in columns_protein[:10]:
        name = name.split(';')[0]
        id = get_protein_id_by_name(name, string_df_info, string_df_alias)
        print(name, id)

    for gene in gene_entrez['gene_name'][:10]:
        string_id = get_protein_id_by_name(gene, string_df_info, string_df_alias, absolute_match=False)
        print(gene, string_id)

def run_test_get_protein_interactors():

    import pickle
    with open('data\protein-interaction\STRING\string_df.pkl', 'rb') as f:
        string_df = pickle.load(f)
        string_df_info = pickle.load(f)
        string_df_alias = pickle.load(f)

    with open('data/proteomic-expression/goncalves-2022-cell/goncalve_proteome_fillna.pkl', 'rb') as f:
        full_protein_matrix = pickle.load(f)
        sin_peptile_exclusion_matrix = pickle.load(f)
        goncalve_cell_line_info = pickle.load(f)

    # import CCLE gene expression data using pickle

    with open('data/gene-expression/CCLE_Public_22Q2/ccle_expression.pkl', 'rb') as f:
        gene_entrez = pickle.load(f)
        ccle = pickle.load(f)

    # import CCLE sample info data using pickle

    with open('data/gene-expression/CCLE_Public_22Q2/ccle_sample_info.pkl', 'rb') as f:
        ccle_sample_info = pickle.load(f)

    columns_protein = sin_peptile_exclusion_matrix.columns

    test_id = get_protein_id_by_name('HSP90AA1', string_df_info, string_df_alias)
    print(test_id)

    interactors = get_protein_interactors(test_id, string_df)
    print(interactors)
    print(len(interactors))
    for ii in interactors[:5]:
        print(get_protein_name_by_id(ii, string_df_info, 'preferred_name'))

    interactors = get_protein_interactors(test_id, string_df, score_threshold=700)
    print(interactors)
    print(len(interactors))
    for ii in interactors[:5]:
        print(get_protein_name_by_id(ii, string_df_info, 'preferred_name'))

def run_test_get_protein_name_by_id():
    import pickle
    with open('data\protein-interaction\STRING\string_df.pkl', 'rb') as f:
        string_df = pickle.load(f)
        string_df_info = pickle.load(f)
        string_df_alias = pickle.load(f)

    with open('data/proteomic-expression/goncalves-2022-cell/goncalve_proteome_fillna.pkl', 'rb') as f:
        full_protein_matrix = pickle.load(f)
        sin_peptile_exclusion_matrix = pickle.load(f)
        goncalve_cell_line_info = pickle.load(f)

    # import CCLE gene expression data using pickle

    with open('data/gene-expression/CCLE_Public_22Q2/ccle_expression.pkl', 'rb') as f:
        gene_entrez = pickle.load(f)
        ccle = pickle.load(f)

    # import CCLE sample info data using pickle

    with open('data/gene-expression/CCLE_Public_22Q2/ccle_sample_info.pkl', 'rb') as f:
        ccle_sample_info = pickle.load(f)

    columns_protein = sin_peptile_exclusion_matrix.columns

    test_id = get_protein_id_by_name('CDK4', string_df_info, string_df_alias)
    print(test_id)

    import pickle 
    with open('data\protein-interaction\STRING\goncalve_to_string_id_df.pkl', 'rb') as f:
        goncalve_to_string_id_df = pickle.load(f)

    name = get_protein_name_by_id(test_id, goncalve_to_string_id_df, 'goncalve_protein_id', check_field_name='string_protein_id')
    print(name)

    interactors = get_protein_interactors(test_id, string_df)
    print(interactors)
    print(len(interactors))

    gon_ids = [n for n in map(lambda x: get_protein_name_by_id(x, goncalve_to_string_id_df, 
                                                               'goncalve_protein_id', 
                                                               check_field_name='string_protein_id'), interactors) if n is not None]
    
    print(gon_ids)
    print(len(gon_ids))
    

def create_joint_dataset_from_proteome_gdsc(drug_name: str, proteome: pd.DataFrame, gdsc: pd.DataFrame, drug_value: str = 'LN_IC50'):
    drug_dataset = gdsc.loc[gdsc['DRUG_NAME'] == drug_name]
    drug_response_data = drug_dataset[['SANGER_MODEL_ID', drug_value]]
    drug_response_data.set_index('SANGER_MODEL_ID', inplace=True)

    # join the matched_proteome_dataset and the drug_response_data by Sanger_Model_ID (model_id)

    joined_dataset = proteome.join(drug_response_data, how='inner')

    return joined_dataset


def dataset_to_sanger_model_id_from_ccle(df: pd.DataFrame, ccle_info_df: pd.DataFrame):
    '''
    Transforms DepMap_ID based-indices of the dataframe to Sanger_Model_ID
    input: df: pd.DataFrame, must have 'ACH-000000' formatted index as cell line identifiers 
    output: pd.DataFrame, with Sanger_Model_ID as index
    '''
    ccle_IDs = ccle_info_df[['DepMap_ID', 'Sanger_Model_ID']]
    ccle_IDs.set_index('DepMap_ID', inplace=True)

    transformed_df = df.join(ccle_IDs, how='inner')
    # remove rows with NaN for Sanger_Model_ID
    transformed_df.dropna(subset=['Sanger_Model_ID'], inplace=True)
    transformed_df.set_index('Sanger_Model_ID', inplace=True)
    return transformed_df
    

def create_joint_dataset_from_ccle_gdsc2(drug_name: str,
                                         drug_df: pd.DataFrame,
                                         ccle_df: pd.DataFrame,
                                         ccle_info_df: pd.DataFrame,
                                         drug_value: str = 'LN_IC50',
                                         keep_drug_name: bool = False, separate_feature_label: bool = False):

    gdsc2 = drug_df
    ccle = ccle_df
    ccle_sample_info = ccle_info_df

    drug_dataset = gdsc2.loc[gdsc2['DRUG_NAME'] == drug_name]

    drug_response_data = drug_dataset[['SANGER_MODEL_ID', drug_value]]
    id_ccle_info = ccle_sample_info[['Sanger_Model_ID', 'DepMap_ID']].dropna()

    # find the intersection between the cell lines in drug response data and the cell lines in CCLE gene expression data using the Sanger_Model_ID

    celllines = drug_response_data['SANGER_MODEL_ID'].unique()
    celllines = [
        cellline for cellline in celllines if cellline in id_ccle_info['Sanger_Model_ID'].unique()]

    # locate the DepMap_ID of the cell lines in drug response data

    depmap_id = []
    for cellline in celllines:
        depmap_id.append(
            id_ccle_info.loc[id_ccle_info['Sanger_Model_ID'] == cellline]['DepMap_ID'].values[0])

    # construct the gene expression dataframe by finding row names that are in the DepMap_ID list

    matched_gene_expression_dataset = ccle.loc[ccle['CELLLINE'].isin(
        depmap_id)]

    # creating matching training feature and label data, gene expressions are features, drug response ic50 is label
    # extract CELLLINE column from matched_gene_expression_dataset

    matched_cellline = matched_gene_expression_dataset['CELLLINE'].tolist()
    matched_sanger_model_id = []

    # find the Sanger_Model_ID of the matched cell lines

    for cellline in matched_cellline:
        matched_sanger_model_id.append(
            id_ccle_info.loc[id_ccle_info['DepMap_ID'] == cellline]['Sanger_Model_ID'].values[0])

    # join the drug response data and the gene expression data through sanger model id as a medium

    matched_drug_response_data = drug_response_data.loc[drug_response_data['SANGER_MODEL_ID'].isin(
        matched_sanger_model_id)]

    # print(matched_drug_response_data.shape)

    matched_drug_response_data = matched_drug_response_data.set_index(
        'SANGER_MODEL_ID')

    matched_gene_expression_dataset.insert(
        0, 'SANGER_MODEL_ID', matched_sanger_model_id)
    matched_gene_expression_dataset = matched_gene_expression_dataset.set_index(
        'SANGER_MODEL_ID')

    # join the matched_drug_response_data and the matched_gene_expression_dataset

    joined_dataset = matched_drug_response_data.join(
        matched_gene_expression_dataset, how='inner')

    if keep_drug_name:
        joined_dataset.insert(1, 'DRUG_NAME', drug_name)

    if separate_feature_label:
        # feature and label data creation

        # extract the feature data from the joined dataset

        feature_data = joined_dataset.drop(columns=[drug_value])
        feature_data.drop(columns=['CELLLINE'], inplace=True)

        # extract the label data from the joined dataset

        label_data = joined_dataset[drug_value]

        return feature_data, label_data

    return joined_dataset


def create_feature_and_label(df: pd.DataFrame, label_name: str = 'LN_IC50'):
    # extract the feature data from the joined dataset

    feature_data = df.drop(columns=[label_name])

    # if the column 'CELLLINE' is present, drop it
    if 'CELLLINE' in feature_data.columns:
        feature_data.drop(columns=['CELLLINE'], inplace=True)

    # extract the label data from the joined dataset

    label_data = df[label_name]

    return feature_data, label_data

# run_test_get_protein_interactors()

# run_test_get_protein_name_by_id()
