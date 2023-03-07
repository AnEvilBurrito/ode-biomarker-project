import pandas as pd
import numpy as np

### data loading
import pickle

# import GDSC2 drug response data using pickle


def loading_ccle():
    with open('data/gene-expression/CCLE_Public_22Q2/ccle_expression.pkl', 'rb') as f:
        gene_entrez = pickle.load(f)
        ccle = pickle.load(f)
    return ccle, gene_entrez

def loading_ccle_info():
    with open('data/gene-expression/CCLE_Public_22Q2/ccle_sample_info.pkl', 'rb') as f:
        ccle_sample_info = pickle.load(f)
    return ccle_sample_info

def loading_gdsc2():
    with open('data/drug-response/GDSC2/cache_gdsc2.pkl', 'rb') as f:
        gdsc2 = pickle.load(f)
        gdsc2_info = pickle.load(f)
    return gdsc2, gdsc2_info

### functions 

def create_joint_dataset_from_ccle_gdsc(drug_name: str, ccle: pd.DataFrame, ccle_sample_info: pd.DataFrame, gdsc: pd.DataFrame, keep_drug_name: bool = False, separate_feature_label: bool = False): 

    drug_dataset = gdsc.loc[gdsc['DRUG_NAME'] == drug_name]

    drug_response_data = drug_dataset[['SANGER_MODEL_ID', 'LN_IC50']]
    id_ccle_info = ccle_sample_info[['Sanger_Model_ID', 'DepMap_ID']].dropna()

    # find the intersection between the cell lines in drug response data and the cell lines in CCLE gene expression data using the Sanger_Model_ID

    celllines = drug_response_data['SANGER_MODEL_ID'].unique()
    celllines = [cellline for cellline in celllines if cellline in id_ccle_info['Sanger_Model_ID'].unique()]

    # locate the DepMap_ID of the cell lines in drug response data

    depmap_id = []
    for cellline in celllines:
        depmap_id.append(id_ccle_info.loc[id_ccle_info['Sanger_Model_ID'] == cellline]['DepMap_ID'].values[0])

    # construct the gene expression dataframe by finding row names that are in the DepMap_ID list

    matched_gene_expression_dataset = ccle.loc[ccle['CELLLINE'].isin(depmap_id)]

    # creating matching training feature and label data, gene expressions are features, drug response ic50 is label
    # extract CELLLINE column from matched_gene_expression_dataset

    matched_cellline = matched_gene_expression_dataset['CELLLINE'].tolist()
    matched_sanger_model_id = []

    # find the Sanger_Model_ID of the matched cell lines

    for cellline in matched_cellline:
        matched_sanger_model_id.append(id_ccle_info.loc[id_ccle_info['DepMap_ID'] == cellline]['Sanger_Model_ID'].values[0])

    # join the drug response data and the gene expression data through sanger model id as a medium 

    matched_drug_response_data = drug_response_data.loc[drug_response_data['SANGER_MODEL_ID'].isin(matched_sanger_model_id)]

    # print(matched_drug_response_data.shape)

    matched_drug_response_data = matched_drug_response_data.set_index('SANGER_MODEL_ID')

    matched_gene_expression_dataset.insert(0, 'SANGER_MODEL_ID', matched_sanger_model_id)
    matched_gene_expression_dataset = matched_gene_expression_dataset.set_index('SANGER_MODEL_ID')

    # join the matched_drug_response_data and the matched_gene_expression_dataset

    joined_dataset = matched_drug_response_data.join(matched_gene_expression_dataset, how='inner')

    if keep_drug_name:
        joined_dataset.insert(1, 'DRUG_NAME', drug_name)
    
    if separate_feature_label:
        # feature and label data creation

        # extract the feature data from the joined dataset

        feature_data = joined_dataset.drop(columns=['LN_IC50'])
        feature_data.drop(columns=['CELLLINE'], inplace=True)

        # extract the label data from the joined dataset

        label_data = joined_dataset['LN_IC50']

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