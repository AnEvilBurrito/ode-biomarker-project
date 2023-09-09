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


    
def pipeline_func(X_train, y_train, **kwargs):
    
    X_transformed, y_transformed = transform_impute_by_zero(X_train, y_train)
    selected_features, scores = f_regression_select(X_transformed, y_transformed, k=100)
    # selected_features, scores = mrmr_select_fcq(X_transformed, y_transformed, K=10, return_index=False)
    selected_features, X_selected = select_preset_features(X_transformed, y_transformed, selected_features)
    model = SVR()
    model.fit(X_selected, y_transformed)
    return {'model': model, 'selected_features': selected_features, 'scores': scores}

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
            'feature_importance': (pipeline_components['selected_features'], pipeline_components['scores'])}



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

    print(f'Newly shuffled columns: {X.columns[:10]}')

    powerkit = Powerkit(X, y) 
    powerkit.add_condition('test', True, pipeline_func, {}, eval_func, {})


    rngs, total_df, meta_df = powerkit.run_until_consensus('test', n_jobs=1, abs_tol=0.000001, 
                                                        rel_tol=0.000001, max_iter=50,
                                                        verbose=True, verbose_level=1, 
                                                        return_meta_df=True, crunch_factor=1)
    
    # save results
    total_df.to_pickle('total_df.pkl')
    meta_df.to_csv('meta_df.csv')
    
    # save rngs
    with open('rngs.pkl', 'wb') as f:
        pickle.dump(rngs, f)