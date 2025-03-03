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
            'y_pred': y_pred, # for plotting purposes
            'y_test': y_test, 
            }