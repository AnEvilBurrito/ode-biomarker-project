from sklearn.metrics import mean_squared_error, r2_score
import os
import sys

path = os.getcwd()
# find the string 'project' in the path, return index
index_project = path.find('project')
# slice the path from the index of 'project' to the end
project_path = path[:index_project+7]
# set the working directory
sys.path.append(project_path)

from tqdm import tqdm
from toolkit import *
# import pipeline from sci-kit learn

'''
Overarching Pipeline Function 
'''

sample_kwargs = {
    'drug_name': 'drug_name',
    'data_link': 'data_link',
    'drug_database': 'drug_database',
    'feature_database': 'feature_database',
    'target_name': 'target_name',
    'powerkit': 'powerkit',
    'pipeline': 'pipeline',
    'pipeline_args': 'pipeline_args',
    'evaluation_func': 'evaluation_func',
    'evaluation_args': 'evaluation_args',
    'experiment_id': 'experiment_id',
    'random_seeds': 'random_seeds',
    'n_cores': 'n_cores',
}

def run_drug_response_prediction(**kwargs):
    
    ### breakdown of keyword arguments
    
    ## Parameter specific to data access and processing 
    drug_name = kwargs['drug_name']
    data_link = kwargs['data_link']
    drug_database = kwargs['drug_database']
    expression_database = kwargs['feature_database']
    target_name = kwargs['target_name'] # LN_IC50 or AUC
    reorder_index = kwargs['reorder_index']
    reorder_data_index_column = kwargs['reorder_data_index_column']
    
    ## Parameters specific to the machine learning process 
    powerkit = kwargs['powerkit']
    pipeline = kwargs['pipeline']
    pipeline_args = kwargs['pipeline_args']
    
    ## Parameters specific to the evaluation process
    evaluation = kwargs['evaluation_func']
    evaluation_args = kwargs['evaluation_args']
    
    ## Other parameters
    experiment_id = kwargs['experiment_id']
    rngs = kwargs['random_seeds']
    n_cores = kwargs['n_cores'] # number of cores to use for parallel processing, by default set to 1 since this function itself may be parallelized
    
    ### Load data
    
    loading_code = f'generic-{drug_database}-{drug_name}-{target_name}-{expression_database}-{reorder_index}-{reorder_data_index_column}'
    feature_data, label_data = data_link.get_data_using_code(loading_code)
    print(f'Data loaded for code {loading_code} Feature Shape {feature_data.shape} Label Shape {label_data.shape}')
    
    ### Extra Preprocessing Steps 
    # ensure all feature column names are strings
    feature_data.columns = [str(col) for col in feature_data.columns]
    # remove Nan values from the feature data
    feature_data = feature_data.dropna(axis=1)
    # ensure all column names are unique by dropping duplicates
    feature_data = feature_data.loc[:,~feature_data.columns.duplicated()]
    print(f'Feature Shape after preprocessing and dropping duplicates {feature_data.shape}')
    
    ### Run pipeline 
    powerkit.add_condition(drug_name, True, pipeline, pipeline_args, evaluation, evaluation_args)
    df = powerkit.run_pipeline(drug_name, rngs, n_cores, True)
    # add a column for the experiment id
    df['experiment_id'] = experiment_id
    return df 

'''
Powerkit Pipeline Functions - Pipeline and Evaluation
'''
def pipeline_tree_methods(X_train, 
                          y_train, 
                          rng, 
                          model_used, 
                          model_extra_args, 
                          pre_filter=True,
                          pre_filter_size=1000,
                          **kwargs):
    
    # RandomForestRegressor or XGBRegressor at the moment 
    if model_used != 'RandomForestRegressor' and model_used != 'XGBRegressor':
        raise ValueError(f'Model not supported for pipeline_tree_methods, use RandomForestRegressor or XGBRegressor, current model_used param is: {model_used}')
    
    # perform feature selection if pre_filter is True
    if pre_filter:
        selected_features, scores = f_regression_select(X_train, y_train, pre_filter_size)
        _, X_selected = select_preset_features(X_train, y_train, selected_features)
    else:
        X_selected = X_train
    model = get_model_from_string(model_used, **model_extra_args)
    model.fit(X_selected, y_train)
    return {'model': model, 
            'model_type': model_used,
            'train_data': X_train,
            'pre_filter': pre_filter,
            'filtered_features': selected_features if pre_filter else None,
            }


def shap_eval_func(X_test, y_test, pipeline_components=None, **kwargs):
    
    '''
    evaluate the performance of a pipeline through pearson correlation, r2, mse, and 
    feature importance scores using mean absolute SHAP values 
    inputs
        X_test: test set features
        y_test: test set labels
        pipeline_components: dictionary of pipeline components, e.g. {'model': model, 'selected_features': selected_features, 'scores': scores}
    '''
    
    ## evaluation of model performance using test set
    X_test, y_test = transform_impute_by_zero_to_min_uniform(X_test, y_test)
    if pipeline_components['filtered_features'] is None: 
        X_selected = X_test
    else:
        _, X_selected = select_preset_features(X_test, y_test, pipeline_components['filtered_features'])
    y_pred = pipeline_components['model'].predict(X_selected)
    # assess performance by pearson correlation
    corr, p_vals = pearsonr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
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
    return {'model_used': pipeline_components['model_type'],
            'model_performance': corr, 
            'pearson_p_vals': p_vals, 
            'r_squared': r2,
            'mse': mse,
            'feature_importance': (features, scores),
            'important_features': features, 
            'feature_scores': scores,
            }