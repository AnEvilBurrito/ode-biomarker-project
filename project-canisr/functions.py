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

'''
Overarching Pipeline Function 
'''

def run_all_drug_pairs_1(**kwargs):
    
    ### breakdown of keyword arguments
    
    ## Parameter specific to data access and processing 
    
    drug_name = kwargs['drug_name']
    data_link = kwargs['data_link']
    drug_database = kwargs['drug_database']
    expression_database = kwargs['feature_database']
    target_name = kwargs['target_name'] # LN_IC50 or AUC
    
    ## Parameters specific to the machine learning process 
    
    powerkit = kwargs['powerkit']
    pipeline = kwargs['pipeline']
    model = kwargs['model']
    
    ## Parameters specific to the evaluation process
    
    evaluation = kwargs['evaluation_func']
    
    ## Other parameters
    
    experiment_id = kwargs['experiment_id']
    
    ### Load data
    
    loading_code = f'generic-gdsc-2-{drug_name}-{target_name}-ccle_protein_expression-true-Cell_Line'
    feature_data, label_data = data_link.get_data_using_code(loading_code)
    print(f'Data loaded for code {loading_code} Feature Shape {feature_data.shape} Label Shape {label_data.shape}')
    
    ### Run pipeline 
    
    powerkit.add_condition(drug_name, False, pipeline, {}, evaluation, {})


'''
Powerkit Pipeline Functions - Pipeline and Evaluation
'''
def pipeline_tree_methods(X_train, y_train, rng, model, **kwargs):
    
    X_transformed, y_transformed = transform_impute_by_zero(X_train, y_train)
    pass 
    return {'model': '', 
            'model_type': '', 
            'train_data': X_train,
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
    return {'model_performance': corr, 
            'p_vals': p_vals, 
            'feature_importance': (features, scores),
            'filter_score': pipeline_components['scores'],
            }