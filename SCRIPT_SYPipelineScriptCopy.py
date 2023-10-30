from toolkit import * 
import DataFunctions as utils 

import os
import sys

from copy import deepcopy

from sklearn.svm import SVR
from sklearn.feature_selection import f_regression

# import variance threshold
from sklearn.feature_selection import VarianceThreshold

from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

    
def pipeline_func(X_train, y_train, rng, use_mrmr=False, pre_select_size=100, wrapper_select_size=10,
                  **kwargs):
    
    X_transformed, y_transformed = transform_impute_by_zero_to_min_uniform(X_train, y_train)
    
    # debug feature selection
    # print(X_transformed.shape, y_transformed.shape)
    
    
    # removing features with zero variance to avoid division by zero in f_regression
    threshold_selector = VarianceThreshold(threshold=0.0)
    threshold_selector.fit(X_transformed)
    threshold_columns = X_transformed.columns[threshold_selector.get_support()]
    X_transformed = X_transformed[threshold_columns]
    
    # print(X_transformed.shape, y_transformed.shape, "after variance threshold")
    
    # preliminary feature selection
    # print('Here! @after transform_impute_by_zero_to_min_uniform')
    if use_mrmr:
        selected_features, scores = mrmr_select_fcq(X_transformed, y_transformed, K=pre_select_size, return_index=False)
    else:
        selected_features, scores = f_regression_select(X_transformed, y_transformed, k=pre_select_size)
        
    # print('Here! @after feature selection')
    
    selected_features, X_selected = select_preset_features(X_transformed, y_transformed, selected_features)
    # tuning hyperparameters
    best_params, best_fit_score_hyperp, hp_results = hypertune_svr(X_selected, y_transformed, cv=5)
    tuned_model = SVR(**best_params)
    
    # print('Here! @after hyperparameter tuning')
    
    
    # given selected_features and scores, select the highest scoring features
    hi_feature = selected_features[np.argmax(scores)]
    # use wrapper method to select features
    wrapper_features, wrapper_scores = greedy_feedforward_select(X_selected, y_transformed, wrapper_select_size, tuned_model, 
                                                                 start_feature=hi_feature,cv=5, verbose=0)
    
    # print('Here! @after wrapper feature selection')
    
    
    _, X_wrapper_selected = select_preset_features(X_selected, y_transformed, wrapper_features)
    tuned_model.fit(X_wrapper_selected, y_transformed)
    return {'model': tuned_model, 
            'selected_features': wrapper_features, 
            'scores': wrapper_scores,
            'hp_results': hp_results, 
            'best_params': best_params, 
            'best_fit_score_hyperp': best_fit_score_hyperp,
            'prelim_selected_features': selected_features,
            'prelim_scores': scores}

def eval_func(X_test, y_test, pipeline_components=None, **kwargs):
    
    '''
    example function to evaluate the performance of a pipeline
    inputs
        X_test: test set features
        y_test: test set labels
        pipeline_components: dictionary of pipeline components, e.g. {'model': model, 'selected_features': selected_features, 'scores': scores}
    '''
    X_test, y_test = transform_impute_by_zero_to_min_uniform(X_test, y_test)
    _, X_selected = select_preset_features(X_test, y_test, pipeline_components['selected_features'])
    y_pred = pipeline_components['model'].predict(X_selected)
    # assess performance by pearson correlation
    corr, p_vals = pearsonr(y_test, y_pred)

    # at the end, return a dictionary of all the information you want to return
    return {'model_performance': corr, 'p_vals': p_vals, 
            'feature_importance': (pipeline_components['selected_features'], pipeline_components['scores']),
            'hp_results': pipeline_components['hp_results'],
            'best_params': pipeline_components['best_params'],
            'best_fit_score_hyperp': pipeline_components['best_fit_score_hyperp'],
            'prelim_selected_features': pipeline_components['prelim_selected_features'],
            'prelim_scores': pipeline_components['prelim_scores']
            }



if __name__ == "__main__": 
    
    ### --- Data Loading Section
    from PathLoader import PathLoader
    path_loader = PathLoader('data_config.env', 'current_user.env')
    from DataLink import DataLink
    data_link = DataLink(path_loader, 'data_codes.csv')
    
    print('Loading data..')
    
    loading_code = 'ccle-gdsc-1-Palbociclib-LN_IC50'
    feature_data, label_data = data_link.get_data_using_code(loading_code)
    print(f'Data loaded for code {loading_code}')
    
    ### --- Result Saving Configuration 
    
    # --- creating folder name and path
    folder_name = 'SYPipelineScriptCopy' # always take the file name of the script after '_'
    
    if not os.path.exists(f'{path_loader.get_data_path()}data/results/{folder_name}'):
        os.makedirs(f'{path_loader.get_data_path()}data/results/{folder_name}')
    
    file_save_path = f'{path_loader.get_data_path()}data/results/{folder_name}/'
    
    ### --- Powerkit running configurations
    powerkit = Powerkit(feature_data, label_data) 
    
    condition = 'SY_test' # running condition name, can be multiple conditions here
    powerkit.add_condition(condition, True, pipeline_func, {}, eval_func, {})
    
    condition2 = 'SY_testMRMR' 
    powerkit.add_condition(condition2, True, pipeline_func, {'use_mrmr': True}, eval_func, {})
    
    params_profile = {'n_jobs': 1, 
                      'abs_tol': 0.001, 
                      'rel_tol': 0.0001, 
                      'max_iter': 50, 
                      'verbose': True,
                      'verbose_level': 1,
                      'return_meta_df': True,
                      'crunch_factor': 1}

    for condition in [condition, condition2]:
    
        print(f'Running powerkit for condition {condition}..')
        rngs, total_df, meta_df = powerkit.run_until_consensus(condition, **params_profile)  
        # --- actual saving of results for specific conditions 
        quick_save_powerkit_results(total_df, meta_df, rngs, condition, file_save_path)