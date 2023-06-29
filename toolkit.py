# package specific imports
from typing import Literal
import Visualisation as vis
import matplotlib.pyplot as plt

## python imports
import pickle
import logging, sys # for logging
from joblib import Parallel, delayed # for parallel processing

# external imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# import random forest regression model
from sklearn.ensemble import RandomForestRegressor

# import support vector machine regression model
from sklearn.svm import SVR

# import elastic net regression model
from sklearn.linear_model import ElasticNet

# import simple mlp regression model
from sklearn.neural_network import MLPRegressor

# import xgb regression model
from xgboost import XGBRegressor

# import k nearest neighbors regression model
from sklearn.neighbors import KNeighborsRegressor

## feature selection
# import feature selection
from sklearn.feature_selection import SelectKBest, f_regression
import shap 
from sklearn.base import BaseEstimator, TransformerMixin


## validation
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

class FirstQuantileImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()
        self.quantile_ = None

    def fit(self, X, y=None):
        self.quantile_ = X.quantile(0.25)
        return self

    def transform(self, X, y=None, return_df=False):
        q = self.quantile_
        for col in X.columns:

            hi_bound = q[col]

            if np.isnan(hi_bound):
                hi_bound = q.dropna().min()

            vals = np.random.uniform(0, hi_bound, size=X[col].isna().sum())
            X.loc[X[col].isna(), col] = vals
        
        if return_df:
            return X
        return X.values
    

def get_model_from_string(model_name, **kwargs):
    if model_name == 'ElasticNet':
        return ElasticNet(**kwargs)
    elif model_name == 'RandomForestRegressor':
        return RandomForestRegressor(**kwargs)
    elif model_name == 'SVR':
        return SVR(**kwargs)
    elif model_name == 'MLPRegressor':
        return MLPRegressor(**kwargs)
    elif model_name == 'XGBRegressor':
        return XGBRegressor(**kwargs)
    elif model_name == 'KNeighborsRegressor':
        return KNeighborsRegressor(**kwargs)
    else:
        raise ValueError(f'{model_name} is not supported')
    
def get_shap_values(model, model_str, train_data, test_data):
    if model_str == 'RandomForestRegressor':
        explainer = shap.TreeExplainer(model, train_data)
    elif model_str == 'ElasticNet':
        explainer = shap.LinearExplainer(model, train_data)
    elif model_str == 'XGBRegressor':
        explainer = shap.TreeExplainer(model, train_data)
    elif model_str == 'MLPRegressor':
        explainer = shap.DeepExplainer(model, train_data)
    else:
        explainer = shap.KernelExplainer(model.predict, train_data)
    shap_values = explainer.shap_values(test_data)
    return shap_values

def get_network_stat_features(X_train, y_train, X_test, nth_degree_neighbours, max_gene_target_disance, statistical_filter_size):
    network_features = nth_degree_neighbours[max_gene_target_disance]
    # perform feature selection on the training set
    selector = SelectKBest(f_regression, k=statistical_filter_size)
    selector.fit(X_train[network_features], y_train)
    # get the selected features
    selected_features = X_train[network_features].columns[selector.get_support()]
    sel_train, sel_test = X_train[selected_features], X_test[selected_features]
    return selected_features, sel_train, sel_test

def get_random_features(X_train, y_train, X_test, selection_size):
    random_features = np.random.choice(X_train.columns, selection_size, replace=False)
    sel_train, sel_test = X_train[random_features], X_test[random_features]
    return random_features, sel_train, sel_test

def get_all_features(X_train, y_train, X_test):
    sel_train, sel_test = X_train, X_test
    return None, sel_train, sel_test

def get_preset_features(X_train, y_train, X_test, preset_features):
    sel_train, sel_test = X_train[preset_features], X_test[preset_features]
    return preset_features, sel_train, sel_test

def impute_by_first_quantile(X_train, y_train, X_test):
    # fit the imputer
    imputer = FirstQuantileImputer()
    imputer.fit(X_train)
    # transform the data
    X_train = imputer.transform(X_train, return_df=True)
    imputer = FirstQuantileImputer()
    imputer.fit(X_test)
    X_test = imputer.transform(X_test, return_df=True)
    return X_train, y_train, X_test

def impute_by_zero(X_train, y_train, X_test):
    X_train = X_train.fillna(0)
    return X_train, y_train, X_test


def run_single_test(condition,
                    condition_to_get_feature_importance,
                    matched_function,
                    extra_arg,
                    model_str,
                    single_model_hyperparameters,
                    rng,
                    feature_data, label_data,
                    cv_split_size,
                    max_feature_save_size,
                    verbose=False):
    
    if verbose:
        print(f'running {model_str} with seed {rng} under {condition} conditions')

    X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=cv_split_size, random_state=rng)

    selected_features, sel_train, sel_test = matched_function(X_train, y_train, X_test, *extra_arg)
    model = get_model_from_string(model_str, **single_model_hyperparameters)
    model.fit(sel_train, y_train)
    y_pred = model.predict(sel_test)
    score = mean_squared_error(y_test, y_pred)
    corr, p_val = pearsonr(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    shap_values = None        
    if condition_to_get_feature_importance:
        shap_values = get_shap_values(model, model_str, sel_train, sel_test)

    if verbose:
        print(f'--- result: prediction correlation {corr}, p-value {p_val}, r-squared {r_squared}')

    # if sel_train and sel_test are too big, they will not be saved
    if sel_train.shape[1] > max_feature_save_size:
        sel_train = None
    if sel_test.shape[1] > max_feature_save_size:
        sel_test = None

    return [rng, model_str, condition, selected_features, 
                        score, corr, p_val, r_squared, shap_values, 
                        sel_train, sel_test, y_test, y_pred]



def run_bulk_test(conditions_to_test, 
                  conditions_to_get_feature_importance, 
                  matched_functions, 
                  extra_args, 
                  models_used, 
                  models_hyperparameters, 
                  rng_seed_lists, 
                  feature_data, label_data, 
                  cv_split_size, 
                  max_feature_save_size=1000, 
                  n_jobs=1, 
                  verbose=True,
                  save_output=True,
                  bulk_run_tag='bulk_run',
                  output_file_path='output.pkl'
                  ):
    
    '''
    This function examines the performance of different models under different conditions, with different initial seed values.
    It returns a pandas dataframe with the results.
    Many aspects of each model instance is saved, to reduce memory footprint however, the model, training data and sometimes 
    the feature importance are not saved.

    Documentation:
    conditions_to_test: list of strings, each string is a condition to test, e.g. 'all', 'random', 'network'
    conditions_to_get_feature_importance: list of booleans, each boolean indicates whether to get feature importance for the corresponding condition
    matched_functions: list of functions, each function is a function that takes in X_train, y_train, X_test, and returns selected_features, sel_train, sel_test
    extra_args: list of lists, each list is a list of extra arguments to pass to the corresponding function in matched_functions
    models_used: list of strings, each string is a model to use, e.g. 'RandomForestRegressor', 'ElasticNet'
    models_hyperparameters: list of dictionaries, each dictionary is a dictionary of hyperparameters to pass to the corresponding model in models_used
    rng_seed_lists: list of numbers, each number is a random seed to use for the train_test_split function
    feature_data: pandas dataframe, the feature data
    label_data: pandas dataframe, the label data
    cv_split_size: float, the size of the test set relative to the whole dataset
    max_feature_save_size: int, if feature exceeds a certain size, the pipeline will ignore it to save memory
    verbose: boolean, whether to print out progress
    save_output: boolean, whether to save the output
    output_file_path: string, the path to save the output, ignored if save_output is False
    '''

    if n_jobs == 1:

        data_collector = []
        for model_str in models_used:
            for rng in rng_seed_lists:
                for j, condition in enumerate(conditions_to_test):
                    data = run_single_test(condition,
                                        conditions_to_get_feature_importance[j],
                                        matched_functions[j],
                                        extra_args[j],
                                        model_str,
                                        models_hyperparameters[j],
                                        rng,
                                        feature_data, label_data,
                                        cv_split_size,
                                        max_feature_save_size,
                                        verbose=verbose)

                    data_collector.append(data)
    else: 
        # use joblib to parallelize the process
        data_collector = Parallel(n_jobs=n_jobs)(delayed(run_single_test)(condition,
                                        conditions_to_get_feature_importance[j],
                                        matched_functions[j],
                                        extra_args[j],
                                        model_str,
                                        models_hyperparameters[j],
                                        rng,
                                        feature_data, label_data,
                                        cv_split_size,
                                        max_feature_save_size,
                                        verbose=False) 
                                        for model_str in models_used
                                        for rng in rng_seed_lists
                                        for j, condition in enumerate(conditions_to_test))

                
    if verbose:
        print('### All models ran')

    df = pd.DataFrame(data_collector, columns=['rng', 'model', 'exp_condition', 'selected_features',
                                            'mse', 'corr', 'p_val', 'r_squared', 'shap_values', 
                                            'X_train', 'X_test', 'y_test', 'y_pred'])

    output_dict = {bulk_run_tag: df}
    if save_output:
        with open(output_file_path, 'wb') as f:
            pickle.dump(output_dict, f)

        if verbose:
            print('### Results saved')

    
    return df

def get_mean_contribution(df, condition='network_f_regression_selection'):
    # df: dataframe with shap_values, X_train, X_test and a 'exp_condition' columns
    # extract all the shap values, match the feature names and store them in a dataframe

    # for the df, select only the row with the exp_condition column == 'experimental'
    df = df[df['exp_condition'] == condition]

    collector = []
    for shap, x_test in zip(df['shap_values'], df['X_test']):
        # print(shap.shape, cols.shape)
        mean_shap = np.abs(shap).mean(axis=0)
        column_names = x_test.columns
        joint_data = list(zip(column_names, mean_shap))
        # sort the joint data by column names
        joint_data.sort(key=lambda x: x[0])
        collector.append(joint_data)

    # first, create a list of column names

    column_names = [x[0] for x in collector[0]]

    shap_df = pd.DataFrame(collector, columns=column_names)

    # for every cell in the dataframe, keep only the shap value, which is the second element in the tuple

    for col in shap_df.columns:
        shap_df[col] = shap_df[col].apply(lambda x: x[1])

    # sort the dataframe columns by the mean shap values

    shap_df = shap_df.reindex(shap_df.mean().sort_values(ascending=False).index, axis=1)
    shap_df.head()


    # compute the mean shap values for each column

    mean_shap_values = shap_df.mean()
    return mean_shap_values

def get_diff_between_feature_contributions(shap_df1: pd.DataFrame, shap_df2: pd.DataFrame):
    # compute the difference in mean shap values for each feature
    diff = shap_df2 - shap_df1
    diff = diff.sort_values(ascending=False)
    return diff

def get_abs_sum_for_feature_contributions(shap_df):
    return shap_df.abs().sum()

