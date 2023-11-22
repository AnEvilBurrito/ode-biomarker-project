'''
Implements Dawson's first year SHAP pipeline
'''

from toolkit import * 
import DataFunctions as utils 

import os
import sys

from copy import deepcopy

from sklearn.svm import SVR
from sklearn.feature_selection import f_regression
from sklearn.linear_model import ElasticNetCV


# import variance threshold
from sklearn.feature_selection import VarianceThreshold

from sklearn.datasets import make_regression
import pandas as pd
import numpy as np


def genenric_pipeline_func(X_train, y_train, rng, use_mrmr=False, pre_select_size=100, use_model='ElasticNet', **kwargs):
    
    
    pass 

    
def shap_pipeline_func(X_train, y_train, 
                       rng,
                       nth_degree_neighbors, 
                       use_mrmr=False, 
                       pre_select_size=100, 
                       max_gene_target_distance=2,
                       use_model='ElasticNet',
                  **kwargs):
    
    '''
    params:
        X_train: pd.Dataframe | training set features
        y_train: pd.Series | training set labels
        nth_degree_neighbors: list[iterable] | list of nth degree neighbors for the target drug 
        use_mrmr: bool | whether to use mrmr feature selection
        pre_select_size: int | number of features to pre-select in f-regression
        max_gene_target_distance: int, option between 1,2,3 | maximum distance between gene and target to be considered for feature selection, lower the distance, higher the potential biological relevance    
    '''
    
    ## step 1. biological relevance filtering 
    
    network_features, network_X_train = select_network_features(X_train, y_train, nth_degree_neighbors, max_gene_target_distance)
    
    ## step 2. statistical filtering  
    
    # imputing missing values by zero and transforming to uniform distribution
    X_transformed, y_transformed = transform_impute_by_zero_to_min_uniform(X_train, y_train)
    # removing features with zero variance to avoid division by zero in f_regression
    threshold_selector = VarianceThreshold(threshold=0.0)
    threshold_selector.fit(X_transformed)
    threshold_columns = X_transformed.columns[threshold_selector.get_support()]
    X_transformed = X_transformed[threshold_columns]
    
    if use_mrmr:
        selected_features, scores = mrmr_select_fcq(X_transformed, y_transformed, K=pre_select_size, return_index=False)
    else:
        selected_features, scores = f_regression_select(X_transformed, y_transformed, k=pre_select_size)
        
    selected_features, _ = select_preset_features(X_transformed, y_transformed, selected_features)
    
    # assess overlap between network features and selected features
    overlap = set(network_features).intersection(set(selected_features))
    overlap_size = len(overlap)
    # print(f'Overlap between network features and selected features: {overlap_size}')
    overlap_ratio = overlap_size / len(selected_features)
    # print(f'Overlap ratio: {overlap_ratio}')
    
    overlap_features = list(overlap)

    ## step 3. training of model (Elastic Net / Linear Regression / SVR / Random Forest / XGBoost / ANN)
    if use_model == 'SVR':
        # best_params, best_fit_score_hyperp, hp_results = hypertune_svr(X_transformed[overlap_features], y_transformed, cv=5)
        # tuned_model = SVR(**best_params)
        # tuned_model.fit(X_transformed[overlap_features], y_transformed)
        model = SVR()
        model.fit(X_transformed[overlap_features], y_transformed)
    
    if use_model == 'ElasticNet':
        model = ElasticNetCV(cv=5, random_state=rng)
        model.fit(X_transformed[overlap_features], y_transformed)
        
    else: 
        raise ValueError(f'Model {use_model} not supported yet, please choose from SVR or ElasticNet')
    
    ## passing key metrics and results to the evaluation function 
    
    return {'model': model, 
            'model_type': use_model,    
            'selected_features': overlap_features, 
            'scores': scores,
            'overlap_ratio': overlap_ratio,
            'overlap_size': overlap_size,
            'train_data': X_transformed[overlap_features],
            'prelim_selected_features': selected_features,
            }

def shap_eval_func(X_test, y_test, pipeline_components=None, **kwargs):
    
    '''
    example function to evaluate the performance of a pipeline
    inputs
        X_test: test set features
        y_test: test set labels
        pipeline_components: dictionary of pipeline components, e.g. {'model': model, 'selected_features': selected_features, 'scores': scores}
    '''
    
    ## evaluation of model performance using test set
    
    X_test, y_test = transform_impute_by_zero_to_min_uniform(X_test, y_test)
    _, X_selected = select_preset_features(X_test, y_test, pipeline_components['selected_features'])
    y_pred = pipeline_components['model'].predict(X_selected)
    # assess performance by pearson correlation
    corr, p_vals = pearsonr(y_test, y_pred)
    
    ## obtaining SHAP values for each feature, mean absolute SHAP values will 
    ## be used as a way to compute feature importance scores 
    shap_values = get_shap_values(pipeline_components['model'], 
                                  pipeline_components['model_type'],
                                  pipeline_components['train_data'], 
                                  X_selected)
    mean_shap_values = np.abs(shap_values).mean(axis=0)

    ## returning key metrics and results 

    features, scores = X_selected.columns.tolist(), mean_shap_values.tolist()
    # at the end, return a dictionary of all the information you want to return
    return {'model_performance': corr, 
            'p_vals': p_vals, 
            'feature_importance': (features, scores),
            'filter_score': pipeline_components['scores'],
            'overlap_ratio': pipeline_components['overlap_ratio'],
            'overlap_size': pipeline_components['overlap_size'],
            }



if __name__ == "__main__": 
    
    ### --- Data Loading Section
    from PathLoader import PathLoader
    path_loader = PathLoader('data_config.env', 'current_user.env')
    from DataLink import DataLink
    data_link = DataLink(path_loader, 'data_codes.csv')
    
    print('Loading data..')
    
    # load in original ccle data
    loading_code = 'ccle-gdsc-2-Palbociclib-LN_IC50-sin'
    feature_data, label_data = data_link.get_data_using_code(loading_code)
    print(f'Data loaded for code {loading_code}')
    print('feature data shape:', feature_data.shape, 'label data shape:', label_data.shape)
    # load in neighbors
    data_link.load_data_code('palbociclib_neighbours_string', verbose=False)
    neighbour_data = data_link.data_code_database['palbociclib_neighbours_string']
    
    
    # load in dynamic feature data
    dynamic_feature_data, dynamic_label_data = data_link.get_data_using_code('anthony-ode-gdsc-2-Palbociclib-LN_IC50-default')

    print(label_data.head())
    
    ### --- Result Saving Configuration 
    
    # --- creating folder name and path
    folder_name = 'SHAPPipeline' # always take the file name of the script after '_'
    
    if not os.path.exists(f'{path_loader.get_data_path()}data/results/{folder_name}'):
        os.makedirs(f'{path_loader.get_data_path()}data/results/{folder_name}')
    
    file_save_path = f'{path_loader.get_data_path()}data/results/{folder_name}/'
    
    ### --- Powerkit running configurations
    powerkit = Powerkit(feature_data, label_data) 
    
    condition = 'testrun' # running condition name, can be multiple conditions here
    pipeline_params = {
        'nth_degree_neighbors': neighbour_data,
        'use_mrmr': False,
        'pre_select_size': 100,
        'max_gene_target_distance': 2
    }
    powerkit.add_condition(condition, True, shap_pipeline_func, pipeline_params, shap_eval_func, {})

    
    params_profile = {'n_jobs': 1, 
                      'abs_tol': 0.001, 
                      'rel_tol': 0.0001, 
                      'max_iter': 5, 
                      'verbose': True,
                      'verbose_level': 1,
                      'return_meta_df': True,
                      'crunch_factor': 1}

    # rngs, total_df, meta_df = powerkit.run_until_consensus(condition, **params_profile)
    
    # quick_save_powerkit_results(total_df, meta_df, rngs, condition, file_save_path)

    # for condition in [condition, condition2]:
    
    #     print(f'Running powerkit for condition {condition}..')
    #     rngs, total_df, meta_df = powerkit.run_until_consensus(condition, **params_profile)  
    #     # --- actual saving of results for specific conditions 
    #     quick_save_powerkit_results(total_df, meta_df, rngs, condition, file_save_path)