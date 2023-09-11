from toolkit import * 
import DataFunctions as utils 

import os
import sys

from copy import deepcopy

from sklearn.svm import SVR
from sklearn.feature_selection import f_regression

from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

### Loading data


from PathLoader import PathLoader

path_loader = PathLoader('data_config.env', 'current_user.env')

### Load data

import pandas as pd
import pickle

# import GDSC2 drug response data using pickle

with open(f'{path_loader.get_data_path()}data/drug-response/GDSC2/cache_gdsc2.pkl', 'rb') as f:
    gdsc2 = pickle.load(f)
    gdsc2_info = pickle.load(f)
    
# import CCLE gene expression data using pickle

with open(f'{path_loader.get_data_path()}data/gene-expression/CCLE_Public_22Q2/ccle_expression.pkl', 'rb') as f:
    gene_entrez = pickle.load(f)
    ccle = pickle.load(f)

# import CCLE sample info data using pickle

with open(f'{path_loader.get_data_path()}data/gene-expression/CCLE_Public_22Q2/ccle_sample_info.pkl', 'rb') as f:
    ccle_sample_info = pickle.load(f)

# import STRING database using pickle

with open(f'{path_loader.get_data_path()}data/protein-interaction/STRING/string_df.pkl', 'rb') as f:
    string_df = pickle.load(f)
    string_df_info = pickle.load(f)
    string_df_alias = pickle.load(f)


# import proteomic expression
with open(f'{path_loader.get_data_path()}data/proteomic-expression/goncalves-2022-cell/goncalve_proteome_fillna_processed.pkl', 'rb') as f:
    joined_full_protein_matrix = pickle.load(f)
    joined_sin_peptile_exclusion_matrix = pickle.load(f)

# import STRING database using pickle

with open(f'{path_loader.get_data_path()}data/protein-interaction/STRING/string_df.pkl', 'rb') as f:
    string_df = pickle.load(f)
    string_df_info = pickle.load(f)
    string_df_alias = pickle.load(f)

# open STRING to goncalves mapping file

with open(f'{path_loader.get_data_path()}data\protein-interaction\STRING\goncalve_to_string_id_df.pkl', 'rb') as f:
    goncalve_to_string_id_df = pickle.load(f)

# open the cache for neighbourhood calculations

with open(f'{path_loader.get_data_path()}data/protein-interaction/STRING/palbociclib_nth_degree_neighbours.pkl', 'rb') as f:
    nth_degree_neighbours = pickle.load(f)

    
def pipeline_func(X_train, y_train, **kwargs):
    
    X_transformed, y_transformed = transform_impute_by_zero(X_train, y_train)
    # preliminary feature selection
    selected_features, scores = f_regression_select(X_transformed, y_transformed, k=100)
    # selected_features, scores = mrmr_select_fcq(X_transformed, y_transformed, K=10, return_index=False)
    selected_features, X_selected = select_preset_features(X_transformed, y_transformed, selected_features)
    # tuning hyperparameters
    model = SVR()
    best_params, best_fit_score_hyperp, hp_results = hypertune_svr(X_selected, y_transformed, cv=5)
    tuned_model = SVR(**best_params)
    
    # given selected_features and scores, select the highest scoring features
    hi_feature = selected_features[np.argmax(scores)]
    # use wrapper method to select features
    wrapper_features, wrapper_scores = greedy_feedforward_select(X_selected, y_transformed, 10, tuned_model, start_feature=hi_feature,cv=5)
    
    _, X_wrapper_selected = select_preset_features(X_selected, y_transformed, wrapper_features)
    tuned_model.fit(X_wrapper_selected, y_transformed)
    return {'model': tuned_model, 'selected_features': wrapper_features, 'scores': wrapper_scores,
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
    
    # turn X and Y into dataframes
    X, y = make_regression(n_samples=500, n_features=1000, n_informative=10, random_state=1, shuffle=False)

    X = pd.DataFrame(X)
    y = pd.Series(y)

    # turn columns into strings

    X.columns = [str(i) for i in range(X.shape[1])]

    print(f'Original informative columns: {X.columns[:10]}')

    # shuffle columns around for X

    X = X.sample(frac=1, axis=1, random_state=0)

    print(f'Newly shuffled columns (non-informative): {X.columns[:10]}')

    powerkit = Powerkit(X, y) 
    powerkit.add_condition('test', True, pipeline_func, {}, eval_func, {})


    rngs, total_df, meta_df = powerkit.run_until_consensus('test', n_jobs=4, abs_tol=0.001, 
                                                        rel_tol=0.0001, max_iter=100,
                                                        verbose=True, verbose_level=1, 
                                                        return_meta_df=True, crunch_factor=1)
    
    # file save path 
    
    file_save_path = f'{path_loader.get_data_path()}data/results/SYPipelineScriptTest/'
    
    # save results
    total_df.to_pickle('total_df_test.pkl')
    meta_df.to_pickle('meta_df_test.pkl')
    
    # save rngs
    with open('rngs_list_test.pkl', 'wb') as f:
        pickle.dump(rngs, f)