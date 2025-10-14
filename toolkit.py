# package specific imports
from typing import Literal
import matplotlib.pyplot as plt

## python imports
import pickle
import logging, sys # for logging
from typing import Callable
from collections import Counter

from joblib import Parallel, delayed, cpu_count # for parallel processing
from tqdm import tqdm # for progress bar

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
from sklearn.linear_model import ElasticNet, LinearRegression

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

import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import clone
import sklearn_relief as sr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# hyperparameter tuning
from sklearn.model_selection import GridSearchCV

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


class FeatureTransformer:
    def __init__(self):
        '''
        chains a series of transform functions and selection functions together,
        with the assumption that the transform/selection functions are independent of each other
        '''
        self.transform_functions = {}
        self.selection_function = {}

    def add_selection_function(self, name: str, selection_function, function_kwargs: dict = {}):
        self.selection_function[name] = {'selection_function': selection_function, 'args': function_kwargs}

    def get_selection_function(self, name: str):
        return self.selection_function[name]['selection_function']
    
    def get_selection_function_args(self, name: str):
        return self.selection_function[name]['args']
    
    def get_selection_function_names(self):
        return self.selection_function.keys()
    
    def remove_selection_function(self, name: str):
        self.selection_function.pop(name)

    def add_transform_function(self, name: str, transform_function, function_kwargs: dict = {}):
        self.transform_functions[name] = {'transform_function': transform_function, 'args': function_kwargs}
    
    def get_transform_function(self, name: str):
        return self.transform_functions[name]['transform_function']
    
    def get_transform_function_args(self, name: str):
        return self.transform_functions[name]['args']
    
    def get_transform_function_names(self):
        return self.transform_functions.keys()
    
    def remove_transform_function(self, name: str):
        self.transform_functions.pop(name)

    def run_all_transform(self, X_train, y_train):
        '''
        Input:
            X_train: pandas dataframe, training data
            y_train: pandas series, training label
            X_test: pandas dataframe, test data
        Output:
            X_train: pandas dataframe, the training data after all transform functions
            y_train: pandas series, the training label after all transform functions
            X_test: pandas dataframe, the test data after all transform functions
        '''
        for name in self.transform_functions.keys():
            transform_function = self.get_transform_function(name)
            transform_args = self.get_transform_function_args(name)
            X_train, y_train = transform_function(X_train, y_train, **transform_args)
        return X_train, y_train
    
    def run_selection_function(self, X_train, y_train):
        '''
        Input:
            X_train: pandas dataframe, training data
            y_train: pandas series, training label
            X_test: pandas dataframe, test data
        Output:
            selected_features: list of strings, the selected features, selected features should only be based on X_train
                Strings are used instead of indices because it guarantees subset selection while maintain stability as 
                the indices of the features may change after an feature selection function 
            sel_train: pandas dataframe, the training data after feature selection
            sel_test: pandas dataframe, the test data after feature selection
        '''
        for name in self.selection_function.keys():
            selection_function = self.get_selection_function(name)
            selection_args = self.get_selection_function_args(name)
            selected_features, sel_train = selection_function(X_train, y_train, **selection_args)
        return selected_features, sel_train
    
    def run(self, X_train, y_train):
        '''
        Simply run all the transform functions and selection functions
        '''
        X_train, y_train = self.run_all_transform(X_train, y_train)
        selected_features, sel_train = self.run_selection_function(X_train, y_train)
        return selected_features, sel_train
    
class Powerkit:

    def __init__(self, feature_data: pd.DataFrame, label_data: pd.Series) -> None:
        self.feature_data = feature_data
        self.label_data = label_data

        self.cv_split_size = 0.1
        self.conditions = {} # dict of dict obj, dict has the following structure: 
                             # {
                             # 'condition': str, 
                             # 'condition_to_get_feature_importance': bool, 
                             # 'pipeline_function': Callable, 
                             # 'pipeline_args': dict, 
                             # 'eval_function': Callable, 
                             # 'eval_args': dict
                             # }

    def add_condition(self, condition: str, get_importance: bool, pipeline_function: Callable, pipeline_args: dict, eval_function: Callable, eval_args: dict):
        # if condition already exists, raise error
        if condition in self.conditions.keys():
            raise ValueError(f'condition {condition} already exists')
        self.conditions[condition] = {'condition': condition, 
                                    'condition_to_get_feature_importance': get_importance, 
                                    'pipeline_function': pipeline_function, 
                                    'pipeline_args': pipeline_args, 
                                    'eval_function': eval_function, 
                                    'eval_args': eval_args}
        
    def remove_condition(self, condition: str):
        # if condition does not exist, raise error
        if condition not in self.conditions.keys():
            raise ValueError(f'condition {condition} does not exist')
        self.conditions.pop(condition)

    def generate_rng_list(self, n_rng):
        rng_list = np.random.randint(0, 100000, size=n_rng)
        return rng_list

    def _abstract_run_single(self, 
                        condition: str,
                        condition_to_get_feature_importance: bool,
                        rng: int,
                        pipeline_function: Callable,
                        pipeline_args: dict, 
                        eval_function: Callable,
                        eval_args: dict,
                        verbose: bool = False
                        ):
        
        '''
        enforces only on how the data is split and a generic df structure with 
        the first column being 'condition'. 

        The rest of the columns are up to the methods `pipeline_function` and `eval_function` to decide. They effectively make up the actual condition and additional return values.

        pipeline function takes in the following arguments:
            X_train: pandas dataframe, training data
            y_train: pandas series, training label
            pipeline_args: dict, the rest of the arguments

        it's possible that eval_function require information from pipeline_function. Usually it is a trained model. 
        In which case the pipeline_function must return a dict variable called `pipeline_components` 
        which contains the information needed for eval_function
            
        eval_function takes in the following arguments:
            X_test: pandas dataframe, test data
            y_test: pandas series, test label
            eval_args: dict | None, the rest of the arguments
            pipeline_components: dict | None, the information needed for eval_function, if not needed, eval_info is None

        things should be returned for each run:
            feature importance: a tuple, of (feature_name, score) for each feature which represents the relative importance of the feature   
            model performance [OPTIONAL]: measured either by accuracy, correlation, etc, depending on the eval_function, if not provided, return None. 
                model performance can be used to calculated an adjusted feature importance metric, e.g. by multiplying the feature importance by the model performance. It can also be used as a proxy to evaluate the generalizability of the model.

        final return value:
            results: dict = {'rng': rng, 'condition': condition, 'feature_importance': feature_importance, 'model_performance': model_performance}

        '''
        # initialize the final returns
        final_returns = {}
        final_returns['rng'] = rng
        final_returns['condition'] = condition
        
        # split the data and go through the pipeline
        X_train, X_test, y_train, y_test = train_test_split(self.feature_data, self.label_data, test_size=self.cv_split_size, random_state=rng)
        pipeline_comps = pipeline_function(X_train, y_train, rng, **pipeline_args)
        eval_returns = eval_function(X_test, y_test, pipeline_components=pipeline_comps, **eval_args)
        
        # update the final returns
        final_returns.update(eval_returns)
        if not condition_to_get_feature_importance:
            final_returns.pop('feature_importance')
        
        return final_returns
    
    def _abstract_run(self, rng_list: list, n_jobs: int, verbose: bool = False, conditions=None):
        
        if conditions is None:
            conditions = self.conditions
        
        if n_jobs == 1:
            # use normal loop syntax for verbose printing
            data_collector = []
            for rng in tqdm(rng_list, disable=not verbose):
                for condition in conditions:
                    data = self._abstract_run_single(condition,
                                                    self.conditions[condition]['condition_to_get_feature_importance'],
                                                    rng,
                                                    self.conditions[condition]['pipeline_function'],
                                                    self.conditions[condition]['pipeline_args'],
                                                    self.conditions[condition]['eval_function'],
                                                    self.conditions[condition]['eval_args'],
                                                    verbose=verbose
                                                    )
                    data_collector.append(data)
        else:
            # use joblib to parallelize the process, disable verbose printing
            data_collector = Parallel(n_jobs=n_jobs)(delayed(self._abstract_run_single)(condition,
                                                    self.conditions[condition]['condition_to_get_feature_importance'],
                                                    rng,
                                                    self.conditions[condition]['pipeline_function'],
                                                    self.conditions[condition]['pipeline_args'],
                                                    self.conditions[condition]['eval_function'],
                                                    self.conditions[condition]['eval_args'],
                                                    verbose=False
                                                    ) 
                                                    for rng in rng_list
                                                    for condition in conditions)
            
        df = pd.DataFrame(data_collector)
        return df
    
    def run_all_conditions(self, rng_list: list, n_jobs: int, verbose: bool = False):
        df = self._abstract_run(rng_list, n_jobs, verbose)
        return df
    
    def run_selected_condition(self, condition: str, rng_list: list, n_jobs: int, verbose: bool = False):
        if condition not in self.conditions.keys():
            raise ValueError(f'condition {condition} does not exist')
        df = self._abstract_run(rng_list, n_jobs, verbose, conditions=[condition])
        return df
    
    def get_mean_contribution(self, df, condition, col_name='feature_importance', adjust_for_accuracy=False, accuracy_col_name='model_performance', **kwargs):
        '''
        Use adjust_for_accuracy ONLY when the distribution of the model performance is similar to that of the feature importance
        '''
        
        # filter the dataframe by condition
        df = df[df['condition'] == condition]
        
        feature_importance = df[col_name]
        rngs = df['rng']
        accuracies = df[accuracy_col_name]

        data_collector = []

        # for each row in the feature importance column, append tuple (feature_name, score) to a list
        accuracy_tuples = []
        for fi_row, rng_row, accuracies in zip(feature_importance, rngs, accuracies):
            for feature_name, score in zip(fi_row[0], fi_row[1]):
                data_collector.append({'iteration_no': rng_row, 'feature_names': feature_name, 'scores': score})
                if adjust_for_accuracy:
                    accuracy_tuples.append((feature_name, accuracies))

        # convert the list to a dataframe
        feature_importance_df = pd.DataFrame(data_collector)

        # if adjust_for_accuracy is False, set accuracy_tuples to None
        if not adjust_for_accuracy:
            accuracy_tuples = None

        # calculate the mean contribution for each feature
        contribution = get_mean_contribution_general(feature_importance_df, adjust_for_accuracy=adjust_for_accuracy, accuracy_scores=accuracy_tuples, **kwargs)
        
        return contribution
            
            
    
    def run_until_consensus(self, condition: str, 
                            rel_tol: float = 0.01, abs_tol: float = 0.001, max_iter: int = 100, 
                            use_std: bool = False,
                            feature_importance_col_name: str = 'feature_importance', 
                            model_performance_col_name: str = 'model_performance',
                            n_jobs: int = 1, verbose: bool = False, verbose_level: int = 1, 
                            return_meta_df: bool = False,
                            crunch_factor=1):
        
        '''
        Input: 
            condition: str, the condition to run
            rel_tol: float, the relative tolerance to use for consensus run
            abs_tol: float, the absolute tolerance to use for consensus run
            max_iter: int, the maximum number of iterations to run
            use_std: bool, whether to use standard deviation as the metric for consensus run, if False, use average absolute difference
            n_jobs: int, the number of jobs to use for parallel processing
            verbose: bool, whether to print verbose information
            verbose_level: int, the level of verbose information to print, 0 for no verbose, 1 for basic information, 2 for intermediate information, 3 for all information
            return_meta_df: bool, whether to return the meta dataframe which contains the information for each iteration
        Output:
            rng_list: list of int, the rng list used for consensus run
            total_df: pandas dataframe, the dataframe containing the results for each iteration
            meta_df: pandas dataframe, the dataframe containing the meta information for each iteration, only returned if return_meta_df is True
        '''
    
        rng_list = []

        current_tol = 1e10 
        abs_diff = 1e10
        current_contrib = 0 
        prev_contrib = 0

        meta_results = []
        total_df = pd.DataFrame()
        early_break = False

        while current_tol > rel_tol and abs_diff > abs_tol and len(rng_list) < max_iter:
            
            if early_break: 
                break
            
            n_rngs = (n_jobs if n_jobs != -1 else cpu_count()) * crunch_factor
            rngs = self.generate_rng_list(n_rngs)

            if verbose and verbose_level >= 3:
                print(f'running condition {condition} with rng {rngs}')
            verbose_at_run = False
            if verbose and verbose_level >= 2:
                verbose_at_run = True
            df = self.run_selected_condition(condition, rng_list=rngs, n_jobs=n_jobs, verbose=verbose_at_run)
            
            # create a mini df for each iteration
            for rng in rngs: 
                mini_df = df[df['rng'] == rng]
                # print(mini_df.shape)
            
                if verbose and verbose_level >= 3:
                    print(f'finished running condition {condition} with rng {rng}')
                if mini_df is None:
                    raise ValueError(f'no df is returned for condition {condition}')

                total_df = pd.concat([total_df, mini_df], axis=0)
                if verbose and verbose_level >= 3:
                    print(f'finished concatenating df for condition {condition} with rng {rng}')

                if isinstance(prev_contrib, int):
                    if verbose and verbose_level >= 3:
                        print(f'prev_contrb is 0, setting prev_contrb to current_contrib')
                    prev_contrib = self.get_mean_contribution(total_df, condition, strict_mean=0, col_name=feature_importance_col_name)
                    # strict mean = 0, sum only at the end
                    if verbose and verbose_level >= 1:
                        print(f'prev_contrib: {list(prev_contrib.index[:5])}')
                        print(f'current iteration: {len(rng_list)} current_tol: {current_tol:4f}, abs_diff: {abs_diff:6f}, performance: {df[model_performance_col_name].mean():2f}')
                else:
                    current_contrib = self.get_mean_contribution(total_df, condition, strict_mean=0, col_name=feature_importance_col_name)
                    # print the first five features in one line by converting to list
                    if verbose and verbose_level >= 1:
                        print(f'current_contrib: {list(current_contrib.index[:5])}')
                        
                    diff = prev_contrib.copy()    
                    diff['scores'] = prev_contrib['scores'] - current_contrib['scores']
                    abs_diff = np.abs(diff['scores']).sum()
                    abs_prev = np.abs(prev_contrib['scores']).sum()
                    if verbose and verbose_level >= 3:
                        # print(f'{prev_contrib}, {current_contrib}')
                        print(f'total abs prev: {abs_prev}, total abs current: {abs_diff}')
                    current_tol = 1 - (abs_prev - abs_diff) / abs_prev
                    prev_contrib = current_contrib
                    if verbose and verbose_level >= 1: 
                        print(f'current iteration: {len(rng_list)} current_tol: {current_tol:4f}, abs_diff: {abs_diff:6f}, abs_prev: {abs_prev:2f}, performance: {total_df[model_performance_col_name].mean():2f}')
                    meta_results.append([len(rng_list), current_tol, abs_diff, abs_prev, total_df[model_performance_col_name].mean()])
                
                rng_list.append(rng)
                if current_tol <= rel_tol or abs_diff <= abs_tol or len(rng_list) >= max_iter:
                    early_break = True
                    break
                    
            
        if verbose and verbose_level >= 0: 
            # display in one line 
            print(f'Consensus Run: condition {condition} is done in {len(rng_list)} iterations')

            if current_tol >= rel_tol:
                print(f'Consensus Run under condition {condition} is NOT converged within {rel_tol} relative tolerance')

            if abs_diff >= abs_tol:
                print(f'Consensus Run under condition {condition} is NOT converged within {abs_tol} absolute tolerance')


            if len(rng_list) >= max_iter:
                print(f'WARNING: Consensus Run under condition {condition} is not converged within {max_iter} iterations')
        
        # create a dataframe for meta results
        
        meta_df = pd.DataFrame(meta_results, columns=['iteration', 'current_tol', 'abs_diff', 'abs_prev', 'perf'])

        if return_meta_df:
            return rng_list, total_df, meta_df
        
        return rng_list, total_df

### pipeline functions 
'''

'''

### evaluation functions 
'''

'''

class Toolkit:

    def __init__(self, feature_data: pd.DataFrame, label_data: pd.Series) -> None:

        '''
        Toolkit contains config parameters and the cleaned dataset for the analysis
        '''

        self.feature_data = feature_data
        self.label_data = label_data
        
        self.conditions = [], # each element: string, e.g. 'all', 'random', 'network'
        self.conditions_to_get_feature_importance = [] # each element: bool, True or False
        self.matched_functions = [] # each element: function, the function to use for feature selection
        self.extra_args_for_functions = []

        self.models_used = []
        self.model_identifier = []
        self.model_hyperparameters = []

        self.rng_list = []
        self.cv_split_size = 0.1 
        self.max_feature_save_size = 1000
        self.verbose = False

    def add_condition(self, condition, condition_to_get_feature_importance, matched_function, extra_args_for_function):
        '''
        Input:
            condition: string, the condition to test, e.g. 'all', 'random', 'network'
            condition_to_get_feature_importance: boolean, whether to get feature importance for this condition
            matched_function: function, the function to use for feature selection
                best practice is to use FeatureTransformer to wrap the function, call FeatureTransformer.run()
            extra_args_for_function: tuple, the extra arguments to pass to the function
        Example Usage:
            `toolkit.add_condition('all', False, FeatureTransformer.run(), (,))`
            The above example adds a condition called 'all', feature importance will not be calculated,
            the feature selection function is FeatureTransformer.run(), and the extra arguments are empty 
        '''
        self.conditions.append(condition)
        self.conditions_to_get_feature_importance.append(condition_to_get_feature_importance)
        self.matched_functions.append(matched_function)
        self.extra_args_for_functions.append(extra_args_for_function)

    def add_model(self, model, model_identifer, model_hyperparameters):
        self.models_used.append(model)
        self.model_identifier.append(model_identifer)
        self.model_hyperparameters.append(model_hyperparameters)

    def set_rng_list(self, rng_list):
        self.rng_list = rng_list

    def generate_rng_list(self, n_rng):
        self.rng_list = np.random.randint(0, 100000, size=n_rng)



    def _generic_run_single(self,
                            condition,
                            condition_to_get_feature_importance,
                            matched_function,
                            extra_arg,
                            model_str,
                            single_model_hyperparameters,
                            rng,
                            verbose=False):
        if verbose:
            print(f'running {model_str} with seed {rng} under {condition} conditions')

        X_train, X_test, y_train, y_test = train_test_split(self.feature_data, self.label_data, test_size=self.cv_split_size, random_state=rng)

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
        if sel_train.shape[1] > self.max_feature_save_size:
            sel_train = None
        if sel_test.shape[1] > self.max_feature_save_size:
            sel_test = None

        return [rng, model_str, condition, selected_features, 
                            score, corr, p_val, r_squared, shap_values, 
                            sel_train, sel_test, y_test, y_pred] 

    def _generic_run(self, 
                    conditions_to_test, 
                    conditions_to_get_feature_importance, 
                    matched_functions, 
                    extra_args, 
                    models_used, 
                    models_hyperparameters,
                    rng_list=None,
                    n_jobs=1, 
                    verbose=True,
                    ):
        
        if rng_list is None:
            rng_list = self.rng_list
            if len(self.rng_list) == 0:
                raise ValueError('Toolkit: rng_list is empty, please set it or generate it')
        
        if n_jobs == 1:

            data_collector = []
            for m, model_str in enumerate(models_used):
                for rng in rng_list:
                    for j, condition in enumerate(conditions_to_test):
                        data = self._generic_run_single(condition,
                                            conditions_to_get_feature_importance[j],
                                            matched_functions[j],
                                            extra_args[j],
                                            model_str,
                                            models_hyperparameters[m],
                                            rng,
                                            verbose=verbose)

                        data_collector.append(data)
        else: 
            # use joblib to parallelize the process
            data_collector = Parallel(n_jobs=n_jobs)(delayed(self._generic_run_single)(condition,
                                            conditions_to_get_feature_importance[j],
                                            matched_functions[j],
                                            extra_args[j],
                                            model_str,
                                            models_hyperparameters[m],
                                            rng,
                                            verbose=False) 
                                            for m, model_str in enumerate(models_used)
                                            for rng in rng_list
                                            for j, condition in enumerate(conditions_to_test))

                    
        if verbose:
            print('### All models ran')

        df = pd.DataFrame(data_collector, columns=['rng', 'model', 'exp_condition', 'selected_features',
                                                'mse', 'corr', 'p_val', 'r_squared', 'shap_values', 
                                                'X_train', 'X_test', 'y_test', 'y_pred'])        
        return df

    def run_selected_condition(self, condition, rng_list=None, n_jobs=1, verbose=False):
        
        modified_conditions = []
        modify_index = None 

        for i, c in enumerate(self.conditions):
            if condition in c:
                modified_conditions.append(c)
                modify_index = i
                break 
        
        # if no condition is found, return None
        if len(modified_conditions) == 0:
            print(f"WARNING: no condition is found for {condition}")
            print(f'available conditions are {self.conditions}')
            return None
        
        modified_conditions_to_get_feature_importance = [self.conditions_to_get_feature_importance[modify_index]]
        modified_matched_functions = [self.matched_functions[modify_index]]
        modified_extra_args_for_functions = [self.extra_args_for_functions[modify_index]]

        df = self._generic_run(modified_conditions,
                                modified_conditions_to_get_feature_importance,
                                modified_matched_functions,
                                modified_extra_args_for_functions,
                                self.models_used,
                                self.model_hyperparameters,
                                rng_list=rng_list,
                                n_jobs=n_jobs,
                                verbose=verbose)
        return df

    def run_selected_model(self, model_identifier, rng_list=None, n_jobs=1, verbose=False):
        
        modified_model_identifiers = []
        modified_model_hyperparameters_index = None

        for i, m in enumerate(self.model_identifier):
            if model_identifier in m:
                modified_model_identifiers.append(m)
                modified_model_hyperparameters_index = i
                break

        # if no model is found, return None
        if len(modified_model_identifiers) == 0:
            print(f"WARNING: no model is found for identifier {model_identifier}")
            print(f'Available model identifiers are {self.model_identifier}')
            return None
        
        modified_model_hyperparameters = [self.model_hyperparameters[modified_model_hyperparameters_index]]
        modified_model_used = [self.models_used[modified_model_hyperparameters_index]]

        df = self._generic_run(self.conditions,
                                self.conditions_to_get_feature_importance,
                                self.matched_functions,
                                self.extra_args_for_functions,
                                modified_model_used,
                                modified_model_hyperparameters,
                                rng_list=rng_list,
                                n_jobs=n_jobs,
                                verbose=verbose)
        return df





    def run_selected_condition_and_model(self, condition, model_identifier, rng_list=None, n_jobs=1, verbose=False):
        modified_conditions = []
        modify_index = None 

        for i, c in enumerate(self.conditions):
            if condition in c:
                modified_conditions.append(c)
                modify_index = i
                break 
        
        # if no condition is found, return None
        if len(modified_conditions) == 0:
            print(f"WARNING: no condition is found for {condition}")
            print(f'available conditions are {self.conditions}')
            return None
        
        modified_model_identifiers = []
        modified_model_hyperparameters_index = None

        for i, m in enumerate(self.model_identifier):
            if model_identifier in m:
                modified_model_identifiers.append(m)
                modified_model_hyperparameters_index = i
                break

        # if no model is found, return None
        if len(modified_model_identifiers) == 0:
            print(f"WARNING: no model is found for identifier {model_identifier}")
            print(f'Available model identifiers are {self.model_identifier}')
            return None
        
        modified_model_hyperparameters = [self.model_hyperparameters[modified_model_hyperparameters_index]]
        modified_model_used = [self.models_used[modified_model_hyperparameters_index]]

        modified_conditions_to_get_feature_importance = [self.conditions_to_get_feature_importance[modify_index]]
        modified_matched_functions = [self.matched_functions[modify_index]]
        modified_extra_args_for_functions = [self.extra_args_for_functions[modify_index]]

        df = self._generic_run(modified_conditions,
                                modified_conditions_to_get_feature_importance,
                                modified_matched_functions,
                                modified_extra_args_for_functions,
                                modified_model_used,
                                modified_model_hyperparameters,
                                rng_list=rng_list,
                                n_jobs=n_jobs,
                                verbose=verbose)
        return df


    def run_all(self, rng_list=None, n_jobs=1, verbose=False):
        
        df = self._generic_run(self.conditions,
                                self.conditions_to_get_feature_importance,
                                self.matched_functions,
                                self.extra_args_for_functions,
                                self.models_used,
                                self.model_hyperparameters,
                                rng_list=rng_list,
                                n_jobs=n_jobs,
                                verbose=verbose)
        return df
    
    def run_until_consensus(self, condition: str, 
                            rel_tol: float = 0.01, 
                            abs_tol: float = 0.001,
                            max_iter: int = 100,
                            n_jobs=1, verbose=True, verbose_level=1, return_meta_df=False):

        
        
        rng_list = []

        current_tol = 1e10 
        abs_diff = 1e10
        current_contrib = 0 
        prev_contrib = 0

        meta_results = []
        total_df = pd.DataFrame()

        while current_tol > rel_tol and abs_diff > abs_tol and len(rng_list) < max_iter:
            
            rng = np.random.randint(0, 10000000)
            rng_list.append(rng)

            rng_list_to_run = [rng]
            if verbose and verbose_level >= 3:
                print(f'running condition {condition} with rng {rng}')
            verbose_at_run = False
            if verbose and verbose_level >= 2:
                verbose_at_run = True
            df = self.run_selected_condition(condition, rng_list=rng_list_to_run, n_jobs=n_jobs, verbose=verbose_at_run)
            if verbose and verbose_level >= 3:
                print(f'finished running condition {condition} with rng {rng}')
            if df is None:
                raise ValueError(f'no df is returned for condition {condition}')
            else: 
                total_df = pd.concat([total_df, df], axis=0)
            if verbose and verbose_level >= 3:
                print(f'finished concatenating df for condition {condition} with rng {rng}')

            if isinstance(prev_contrib, int):
                if verbose and verbose_level >= 3:
                    print(f'prev_contrb is 0, setting prev_contrb to current_contrib')
                prev_contrib = get_mean_contribution(total_df, condition, absolute_value=True, strict_mean=0)
                # strict mean = 0, sum only at the end
            else:
                current_contrib = get_mean_contribution(total_df, condition, absolute_value=True, strict_mean=0)
                # print the first five features in one line by converting to list
                if verbose and verbose_level >= 1:
                    print(f'current_contrib: {list(current_contrib.index[:5])}')
                if verbose and verbose_level >= 3:
                    # print(f'{prev_contrib}, {current_contrib}')
                    print(f'total abs prev: {get_abs_sum_for_feature_contributions(prev_contrib)}, total abs current: {get_abs_sum_for_feature_contributions(current_contrib)}')
                diff = get_diff_between_feature_contributions(current_contrib, prev_contrib)
                abs_diff = get_abs_sum_for_feature_contributions(diff)
                abs_prev = get_abs_sum_for_feature_contributions(prev_contrib)
                current_tol = 1 - (abs_prev - abs_diff) / abs_prev
                prev_contrib = current_contrib
                if verbose and verbose_level >= 1: 
                    print(f'current iteration: {len(rng_list)} current_tol: {current_tol:4f}, abs_diff: {abs_diff:6f}, abs_prev: {abs_prev:2f}, corr: {df["corr"].mean():2f}')
                meta_results.append([len(rng_list), current_tol, abs_diff, abs_prev, df['corr'].mean()])
            
        if verbose and verbose_level >= 0: 
            # display in one line 
            print(f'Consensus Run: condition {condition} is done in {len(rng_list)} iterations')

            if current_tol >= rel_tol:
                print(f'Consensus Run under condition {condition} is NOT converged within {rel_tol} relative tolerance')

            if abs_diff >= abs_tol:
                print(f'Consensus Run under condition {condition} is NOT converged within {abs_tol} absolute tolerance')


            if len(rng_list) >= max_iter:
                print(f'WARNING: Consensus Run under condition {condition} is not converged within {max_iter} iterations')
        
        # create a dataframe for meta results
        
        meta_df = pd.DataFrame(meta_results, columns=['iteration', 'current_tol', 'abs_diff', 'abs_prev', 'corr'])

        if return_meta_df:
            return rng_list, total_df, meta_df
        
        return rng_list, total_df

def get_model_from_string(model_name, **kwargs):
    if model_name == 'ElasticNet':
        return ElasticNet(**kwargs)
    if model_name == 'LinearRegression':
        return LinearRegression(**kwargs)
    elif model_name == 'RandomForestRegressor':
        return RandomForestRegressor(**kwargs)
    elif model_name == 'SVR':
        svr = SVR(**kwargs)
        return Pipeline([('scaler', StandardScaler()), ('svr', svr)])
    elif model_name == 'MLPRegressor':
        nn = MLPRegressor(**kwargs)
        return Pipeline([('scaler', StandardScaler()), ('nn', nn)])
    elif model_name == 'XGBRegressor':
        return XGBRegressor(**kwargs)
    elif model_name == 'KNeighborsRegressor':
        return KNeighborsRegressor(**kwargs)
    else:
        raise ValueError(f'{model_name} is not supported')
    
def get_shap_values(model, model_str, train_data, test_data):
    # generic shap value calculation wrapper 
    if model_str == 'RandomForestRegressor':
        explainer = shap.TreeExplainer(model)
    elif model_str == 'ElasticNet' or model_str == 'LinearRegression':
        explainer = shap.LinearExplainer(model, train_data)
    elif model_str == 'XGBRegressor':
        explainer = shap.TreeExplainer(model)
    # TODO: tensorflow error for this one, fix
    # elif model_str == 'MLPRegressor':
    #     # model might be in a pipeline, so get the last step
    #     if hasattr(model, 'named_steps'):
    #         model = model.named_steps['nn']
    #     explainer = shap.DeepExplainer(model, train_data)
    else:
        # if train_data is too large (>50), then sample the data
        if train_data.shape[0] > 50:
            train_data = shap.kmeans(train_data, 50)
        explainer = shap.KernelExplainer(lambda x: model.predict(x), train_data)
    shap_values = explainer.shap_values(test_data)
    return shap_values

### Hyperparameter Tuning of Models
'''
All hyperparameter tuning methods should take in the following arguments:
    X: pandas dataframe | numpy array, the data to perform feature selection on
    y: pandas series | numpy array, the label
    cv: int, the number of folds for cross validation
    n_jobs: int, the number of jobs to run in parallel, usually set to 1

All feature selection methods should return the following:
    params: dict, the best parameters for the model
'''

def hypertune_svr(X: pd.DataFrame, y: pd.Series, cv=5, n_jobs=1):
    '''
    Input:
        X: pandas dataframe, the training data
        y: pandas series, the training label
    Output:
        best_params: dict, the best parameters for the model
    '''

    # define the parameter values that should be searched
    kernel_range = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    # instantiate and fit the grid
    grid = GridSearchCV(SVR(), kernel_range, cv=cv, scoring='r2', n_jobs=n_jobs)
    grid.fit(X, y)

    # view the complete results
    # print(grid.cv_results_)

    # examine the best model
    # print(grid.best_score_)
    # print(grid.best_params_)

    return grid.best_params_, grid.best_score_, grid.cv_results_

def hypertune_ann(X: pd.DataFrame, y: pd.Series, cv=5, n_jobs=1):
    '''
    Input:
        X: pandas dataframe, the training data
        y: pandas series, the training label
    Output:
        best_params: dict, the best parameters for the model
    '''

    # define the parameter values that should be searched
    hidden_layer_sizes_range = [(i,) for i in range(1, 100, 10)]
    hidden_layer_sizes_range += [(i, i) for i in range(1, 100, 10)]
    hidden_layer_sizes_range += [(i, i, i) for i in range(1, 100, 10)]
    learning_rate_range = ['constant', 'invscaling', 'adaptive']
    param_grid = dict(hidden_layer_sizes=hidden_layer_sizes_range,
                      learning_rate=learning_rate_range)

    # instantiate and fit the grid
    grid = GridSearchCV(MLPRegressor(max_iter=10000), param_grid, cv=cv, scoring='r2', n_jobs=n_jobs)
    grid.fit(X, y)

    # view the complete results
    # print(grid.cv_results_)

    # examine the best model
    # print(grid.best_score_)
    # print(grid.best_params_)

    return grid.best_params_, grid.best_score_, grid.cv_results_


### Feature Selection Methods 
'''
All feature selection methods should take in the following arguments:
    X: pandas dataframe | numpy array, the data to perform feature selection on
    y: pandas series | numpy array, the label
    k: int, the number of features to select, -1 if all features should be selected
    *args: extra arguments
All feature selection methods should return the following:
    selected_features: list of strings or list of ints representing the indices of the selected features
    scores: list of floats | ints, the scores associated with each selected feature
'''

def ensemble_percentile_threshold(X: pd.DataFrame, y: pd.Series, k: int, 
                                  methods: list, method_kwargs: list, method_cutoffs_way: list,
                                  alpha=0.05, shuffled_iters=1000, n_jobs=1, stable_seeds=True, verbose=1):
    '''
    Given a set of feature selection methods, select features which has alpha < 0.05 in any of the methods
    when compared to feature score distributions based on randomly shuffled label data. 
    
    Input:
        X: pandas dataframe, the training data
        y: pandas series, the training label
        k: int, the number of features to select, always set to -1, k is not used in this method
        method: list of functions, the list of feature selection methods to use
        method_kwargs: list of dicts, the list of extra arguments for each feature selection method
        method_cutoffs_way: list of strings, "one_way" or "two_way", the way to determine the cutoff for each method
        threshold: float, the threshold to select features scores 
        shuffled_iters: int, the number of iterations to shuffle the label data
        n_jobs: int, the number of jobs to run in parallel
        stable_seeds: bool, whether to use stable seeds for reproducibility, stable seeds simply range from 0 to shuffled_iters
    '''
    
    if len(methods) != len(method_kwargs) or len(methods) != len(method_cutoffs_way):
        raise ValueError('methods, method_kwargs, and method_cutoffs_way should have the same length')
    
    # create a list of shuffled labels
    if stable_seeds:
        random_seeds = list(range(shuffled_iters))
    else: 
        random_seeds = np.random.randint(-100000000, 100000000, size=shuffled_iters)
    shuffled_label_data = [y.sample(frac=1, random_state=seed) for seed in random_seeds]
    
    df_above_threshold_list = []
    i = 0 
    while i < len(methods): 
        method, method_kwarg, method_cutoff_way = methods[i], method_kwargs[i], method_cutoffs_way[i]
        all_scores = get_shuffled_scores(shuffled_label_data, X, method, method_kwarg, n_jobs=n_jobs, verbose=verbose)
        all_scores = [score for score in all_scores if not np.isnan(score)]
        if method_cutoff_way == 'one_way':
            percentile_cutoff = 100 - alpha * 100
            threshold = np.percentile(all_scores, percentile_cutoff)
            # print(f'percentile cutoff: {percentile_cutoff}, threshold: {threshold}')
            selected, scores = method(X, y, X.shape[1], **method_kwarg)
            df = build_dataframe(selected, scores, X)
            df_above_threshold = df[df['Scores'] > threshold]
            df_above_threshold_list.append(df_above_threshold)
        elif method_cutoff_way == 'two_way':
            all_scores = [score for score in all_scores if not np.isnan(score)]
            upper_cutoff_value = 100 - (alpha * 100 / 2)
            lower_cutoff_value = (alpha * 100 / 2)
            upper_threshold = np.percentile(all_scores, upper_cutoff_value)
            lower_threshold = np.percentile(all_scores, lower_cutoff_value)
            # print(f'upper cutoff: {upper_cutoff_value}, upper threshold: {upper_threshold}')
            # print(f'lower cutoff: {lower_cutoff_value}, lower threshold: {lower_threshold}')
            selected, scores = method(X, y, X.shape[1], **method_kwarg)
            df = build_dataframe(selected, scores, X)
            df = df.dropna()
            df_above_threshold = df[df['Scores'] > upper_threshold]
            df_below_threshold = df[df['Scores'] < lower_threshold]
            # join the two dataframes
            df_above_threshold = pd.concat([df_above_threshold, df_below_threshold])
            df_above_threshold_list.append(df_above_threshold)
        i += 1 
    
    union_labels = []
    for df in df_above_threshold_list:
        union_labels.extend(df.index.tolist())
    label_counts = Counter(union_labels)

    scores = [count for _, count in label_counts.items()]
    features_selected = [label for label, _ in label_counts.items()]
    return features_selected, scores


def mrmr_select_fcq(X: pd.DataFrame, y: pd.Series, K: int, verbose=0, return_index=True):

    # ------------ Input
    # X: pandas.DataFrame, features
    # y: pandas.Series, target variable
    # K: number of features to select
    
    # ------------ Output
    # feature_selected[List[Int]]: list of selected features index format 
    # successive_scores[List[Float]]: list of successive scores

    # compute F-statistics and initialize correlation matrix
    F = pd.Series(f_regression(X, y)[0], index = X.columns)
    corr = pd.DataFrame(.00001, index = X.columns, columns = X.columns)

    # initialize list of selected features and list of excluded features
    selected = []
    successive_scores = []
    not_selected = X.columns.to_list()

    # repeat K times
    for i in range(K):
    
        # compute (absolute) correlations between the last selected feature and all the (currently) excluded features
        if i > 0:
            last_selected = selected[-1]
            corr.loc[not_selected, last_selected] = X[not_selected].corrwith(X[last_selected]).abs().clip(.00001)
            
        # compute FCQ score for all the (currently) excluded features
        score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(axis = 1).fillna(.00001)
        
        # find best feature, add it to selected and remove it from not_selected
        best = score.index[score.argmax()]
        successive_scores.append(score.max())
        selected.append(best)
        not_selected.remove(best)

        if verbose == 1: 
            print('Iteration', i+1, 'selected', best, 'score', score.max(), 'remaining', len(not_selected), 'features')
    
    if return_index:
        return [X.columns.get_loc(c) for c in selected], successive_scores

    return selected, successive_scores

def mrmr_select_fcq_fast(X: pd.DataFrame, y: pd.Series, K: int, verbose=0, return_index=False):
    """
    Fast mRMR-FCQ feature selection with further optimizations
    May have minor differences from original but much faster
    """
    n_features = X.shape[1]
    feature_names = X.columns.tolist()
    X_np = X.values
    y_np = y.values
    
    # Precompute everything upfront
    corr_matrix = np.abs(np.corrcoef(X_np, rowvar=False))
    from sklearn.feature_selection import f_regression
    F, _ = f_regression(X_np, y_np)
    
    selected = []
    remaining = set(range(n_features))
    successive_scores = []
    
    for i in range(min(K, n_features)):
        if i == 0:
            # Select most relevant feature
            scores = F
        else:
            # Compute redundancy for each remaining feature
            redundancy = np.zeros(n_features)
            for feat in remaining:
                if i == 1:
                    redundancy[feat] = corr_matrix[feat, selected[0]]
                else:
                    # Use vectorized mean calculation
                    redundancy[feat] = np.mean(corr_matrix[feat, selected])
            
            # FCQ score: relevance / redundancy
            scores = np.divide(F, redundancy, 
                             out=np.full_like(F, -np.inf), 
                             where=redundancy>1e-8)
        
        # Find best among remaining features
        best_score = -np.inf
        best_feature = None
        
        for feat in remaining:
            if scores[feat] > best_score:
                best_score = scores[feat]
                best_feature = feat
        
        if best_feature is None:
            break
            
        selected.append(best_feature)
        remaining.remove(best_feature)
        successive_scores.append(best_score)
        
        if verbose:
            print(f'Iteration {i+1}: selected {feature_names[best_feature]}, score {best_score:.4f}')
    
    if return_index:
        return selected, successive_scores
    else:
        return [feature_names[i] for i in selected], successive_scores

def enet_select(X: pd.DataFrame, y: pd.Series, k: int, **kwargs):
    enet = ElasticNet(**kwargs)
    enet.fit(X,y)
    coef = enet.coef_
    abs_coef = np.abs(coef)
    indices = np.flip(np.argsort(abs_coef), 0)[0:k]
    return indices, coef[indices]
    

def relieff_select(X: pd.DataFrame, y: pd.Series, k: int, n_jobs=1):
    if n_jobs >= 1:
        r = sr.RReliefF(n_features = k, n_jobs = n_jobs)
    else: 
        r = sr.RReliefF(n_features = k)
    r.fit(X.to_numpy(), y.to_numpy())
    feat_indices = np.flip(np.argsort(r.w_), 0)[0:k]
    return feat_indices, r.w_[feat_indices]

def rf_select(X: pd.DataFrame, y: pd.Series, k: int, **kwargs):
    rf = RandomForestRegressor(**kwargs)
    rf.fit(X, y)
    coef = rf.feature_importances_
    indices = np.argsort(coef)[::-1][:k]
    return indices, coef[indices]

def f_regression_select(X: pd.DataFrame, y: pd.Series, k: int, *args):
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X, y)
    # get the selected features
    selected_features = X.columns[selector.get_support()]
    return selected_features, selector.scores_[selector.get_support()]

def mutual_information_select(X: pd.DataFrame, y: pd.Series, k: int, *args):
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X, y)
    # get the selected features
    selected_features = X.columns[selector.get_support()]
    return selected_features, selector.scores_[selector.get_support()]

def pearson_corr_select(X: pd.DataFrame, y: pd.Series, k: int, **kwargs):
    # iterate every column of X, a pd dataframe
    data_collector = []
    for (col_name, col_data) in X.items():
        corr, p_vals = pearsonr(col_data, y)
        data_collector.append((col_name, corr, p_vals))
        
    # sort the data collector by the absolute value of the correlation
    data_collector = sorted(data_collector, key=lambda x: abs(x[1]), reverse=True)
    selected_features = [x[0] for x in data_collector[:k]]
    if kwargs['return_all'] == True: 
        # NOTE that this breaks the pattern of the return value for feature selection functions
        return selected_features, [x[1] for x in data_collector[:k]], [x[2] for x in data_collector[:k]]
    return selected_features, [x[1] for x in data_collector[:k]]


def variance_select(X: pd.DataFrame, y: pd.Series, k: int, *args):
    pass 


def wrapper_rfs_select(X: pd.DataFrame, y: pd.Series, k: int, **kwargs):
    '''
    
    '''
    pass 

def greedy_feedforward_select(X: pd.DataFrame, y: pd.Series, k: int, 
                              model: BaseEstimator, start_feature: str, 
                              cv: int,
                              scoring_method: str='r2', verbose=0):
    '''
    
    '''
    
    selected = []
    scores = []
    
    selected.append(start_feature)
    score = cross_val_score(model, X[selected], y, cv=cv, scoring=scoring_method).mean()
    
    remaining_features = X.columns.to_list()
    remaining_features.remove(start_feature)
    
    while len(selected) < k:
        max_score = -10000
        for f in remaining_features: 
            selected.append(f)
            score = cross_val_score(model, X[selected], y, cv=cv, scoring=scoring_method).mean()
            if score > max_score:
                max_score = score
                max_feature = f
            selected.pop()
        selected.append(max_feature)
        scores.append(max_score)
        remaining_features.remove(max_feature)
        if verbose == 1:
            print(f'Feature Selected: {max_feature}, Score: {max_score}, Feature Size: {len(selected)}')
    
    return selected, scores

def greedy_feedforward_select_sy(X: pd.DataFrame, y: pd.Series, k: int, 
                                model: BaseEstimator, start_feature: str, 
                                cv: int, 
                                verbose=0):
    '''
    
    '''
    
    selected = []
    scores = []
    
    selected.append(start_feature)
    
    # split the data into training and testing in cross validation folds 
    score = 0
    kf = KFold(n_splits=cv)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train[selected], y_train)
        y_pred = model.predict(X_test[selected])
        s, _ = pearsonr(y_test, y_pred)
        score += s
    score /= cv
    scores.append(score)
    
    remaining_features = X.columns.to_list()
    remaining_features.remove(start_feature)
    
    while len(selected) < k:
        max_score = -10000
        for f in remaining_features: 
            selected.append(f)
            kf = KFold(n_splits=cv)
            score = 0
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train[selected], y_train)
                y_pred = model.predict(X_test[selected])
                s, _ = pearsonr(y_test, y_pred)
                score += s
            score /= cv
            
            if score > max_score:
                max_score = score
                max_feature = f
            selected.pop()
        selected.append(max_feature)
        scores.append(max_score)
        remaining_features.remove(max_feature)
        if verbose == 1:
            print(f'Feature Selected: {max_feature}, Score: {max_score}, Feature Size: {len(selected)}')
    
    return selected, scores, model

### Selection functions
'''
All selection functions should take in the following arguments:
    X: pandas dataframe, training data
    y: pandas series, training label
    *args: extra arguments
All selection functions should return the following:
    selected_features: list of strings, the selected features, selected features should only be based on X_train
    selected_X: pandas dataframe, the training data after feature selection
'''

def get_network_stat_features(X_train, y_train, X_test, nth_degree_neighbours, max_gene_target_disance, statistical_filter_size):
    '''
    TODO: Plan for deprecation, this function is not compliant with FeatureTransformer pattern
    WARNING: nests two selection functions together, this is not recommended. 
    '''
    network_features = nth_degree_neighbours[max_gene_target_disance]
    # perform feature selection on the training set
    selector = SelectKBest(f_regression, k=statistical_filter_size)
    selector.fit(X_train[network_features], y_train)
    # get the selected features
    selected_features = X_train[network_features].columns[selector.get_support()]
    sel_train, sel_test = X_train[selected_features], X_test[selected_features]
    return selected_features, sel_train, sel_test

def get_random_features(X_train, y_train, X_test, selection_size):
    '''
    TODO: Plan for deprecation, this function is not compliant with FeatureTransformer pattern
    '''

    random_features = np.random.choice(X_train.columns, selection_size, replace=False)
    sel_train, sel_test = X_train[random_features], X_test[random_features]
    return random_features, sel_train, sel_test

def get_all_features(X_train, y_train, X_test):
    '''
    TODO: Plan for deprecation, this function is not compliant with FeatureTransformer pattern
    '''
    sel_train, sel_test = X_train, X_test
    return None, sel_train, sel_test

def get_preset_features(X_train, y_train, X_test, preset_features):
    '''
    TODO: Plan for deprecation, this function is not compliant with FeatureTransformer pattern
    '''
    sel_train, sel_test = X_train[preset_features], X_test[preset_features]
    return preset_features, sel_train, sel_test


def select_network_features(X_train, y_train, nth_degree_neighbours, max_gene_target_disance):
    
    network_features = nth_degree_neighbours[max_gene_target_disance]
    return network_features, X_train[network_features]

def select_stat_features(X_train, y_train, selection_size):
    '''
    Select based on f-regression
    '''
    # perform feature selection on the training set
    selector = SelectKBest(f_regression, k=selection_size)
    selector.fit(X_train, y_train)
    # get the selected features
    selected_features = X_train.columns[selector.get_support()]
    sel_train = X_train[selected_features]
    return selected_features, sel_train

def select_preset_features(X_train, y_train, preset_features: pd.Index):
    '''
    Select based on preset features
    '''
    sel_train = X_train[preset_features]
    return preset_features, sel_train

def select_random_features(X_train, y_train, selection_size):
    '''
    Select based on random features
    '''
    random_features = np.random.choice(X_train.columns, selection_size, replace=False)
    sel_train = X_train[random_features]
    return random_features, sel_train

### Transforming functions 
'''
All transform functions should take in the following arguments:
    X: pandas dataframe, training data
    y: pandas series, training label
    *args: extra arguments
All transform functions should return the following:
    X: pandas dataframe, the training data after transformation
    y: pandas series, the training label after transformation
'''

def impute_by_first_quantile(X, y):
    '''
    Fixed 
    '''
    # fit the imputer
    imputer = FirstQuantileImputer()
    imputer.fit(X)
    # transform the data
    X = imputer.transform(X, return_df=True)
    return X, y

def impute_by_zero(X_train, y_train, X_test):
    '''
    TODO: Plan for deprecation, this function is not compliant with the Tranforming functions pattern
    '''
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    return X_train, y_train, X_test

def transform_impute_by_zero(X, y):
    X = X.fillna(0)
    return X, y

def transform_impute_by_first_quantile(X, y):
    imputer = FirstQuantileImputer()
    imputer.fit(X)
    X = imputer.transform(X, return_df=True)
    return X, y

def transform_impute_by_zero_to_min_uniform(X, y=None, assumed_absolute_min=0, verbose=0):
    X_imputed = X.copy()
    min_values = X_imputed.min(axis=0)
    # sample from uniform distribution between 0 and minimum value of each column
    # for each column, replace nan with the sampled value
    for col in X_imputed.columns:

        min_ = min_values[col]
        
        # print(f'min for {col}: {min_}')

        if np.isnan(min_):
            min_ = min_values.dropna().min() # if minimum value is nan, use the minimum value of the non-nan values

        vals = np.random.uniform(assumed_absolute_min, min_, size=X_imputed[col].isna().sum())
        # print the number of nan values in each column
        if len(vals) > 0 and verbose > 0:
            print(f'min for {col}: {min_}. Number of nan values in {col}: {X_imputed[col].isna().sum()}, number of values sampled: {len(vals)}, {vals}')
        X_imputed.loc[X_imputed[col].isna(), col] = vals
    
    return X_imputed, y

### Selection functions with imputation built-in 
'''
These functions are also selection functions, but they are not recommended
as they are no longer compliant with the FeatureTransformer pattern.
'''

def impute_with_random_selection(X_train, y_train, X_test, n_features):
    X_train, y_train, X_test = impute_by_first_quantile(X_train, y_train, X_test)
    features, X_train, X_test = get_random_features(X_train, y_train, X_test, n_features)
    return features, X_train, X_test

def impute_with_stat_selection(X_train, y_train, X_test, statistical_filter_size):
    X_train, y_train, X_test = impute_by_first_quantile(X_train, y_train, X_test)
    # perform feature selection on the training set
    selector = SelectKBest(f_regression, k=statistical_filter_size)
    selector.fit(X_train, y_train)
    # get the selected features
    selected_features = X_train.columns[selector.get_support()]
    sel_train, sel_test = X_train[selected_features], X_test[selected_features]
    return selected_features, sel_train, sel_test

def impute_with_network_stat_selection(X_train, y_train, X_test, nth_degree_neighbours, max_gene_target_disance, statistical_filter_size):
    X_train, y_train, X_test = impute_by_first_quantile(X_train, y_train, X_test)
    features, sel_train, sel_test = get_network_stat_features(X_train, y_train, X_test, nth_degree_neighbours, max_gene_target_disance, statistical_filter_size)
    return features, sel_train, sel_test

def impute_with_preset_features(X_train, y_train, X_test, preset_features):
    X_train, y_train, X_test = impute_by_first_quantile(X_train, y_train, X_test)
    features, sel_train, sel_test = get_preset_features(X_train, y_train, X_test, preset_features)
    return features, sel_train, sel_test

### Single machine learning model run function 

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
        for m, model_str in enumerate(models_used):
            for rng in rng_seed_lists:
                for j, condition in enumerate(conditions_to_test):
                    data = run_single_test(condition,
                                        conditions_to_get_feature_importance[j],
                                        matched_functions[j],
                                        extra_args[j],
                                        model_str,
                                        models_hyperparameters[m],
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
                                        models_hyperparameters[m],
                                        rng,
                                        feature_data, label_data,
                                        cv_split_size,
                                        max_feature_save_size,
                                        verbose=False) 
                                        for m, model_str in enumerate(models_used)
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

def get_mean_contribution(df, condition='random', absolute_value=True, strict_mean=0.25, adjust_for_accuracy=False):
    # df: dataframe with shap_values, X_train, X_test and a 'exp_condition' columns
    # extract all the shap values, match the feature names and store them in a dataframe

    # strict mean: feature must be present in at least x% of iterations
    # adjust_for_accuracy: if True, the mean shap value is divided by the accuracy metric of the model 
    # TODO: NOT IMPLEMENTED YET

    # for the df, select only the row with the exp_condition column == 'experimental'
    df = df[df['exp_condition'] == condition]

    all_shap_values = []
    for i in range(df.shape[0]):
        shap_values = df['shap_values'].iloc[i]
        X_test = df['X_test'].iloc[i]
        if absolute_value:
            mean_shap_values = np.abs(shap_values).mean(axis=0)
        else:
            mean_shap_values = shap_values.mean(axis=0)
        mean_shap_df = pd.DataFrame({'mean_shap_values': mean_shap_values, 'feature_names': X_test.columns.tolist()})
        mean_shap_df.sort_values(by='mean_shap_values', ascending=False, inplace=True)
        all_shap_values.append(mean_shap_df)

    # for all shap values, join them together
    all_shap_values_df = pd.concat(all_shap_values)

    # get the occurrence of each feature name
    feature_count = all_shap_values_df.groupby('feature_names').count()

    # group by feature name and compute the mean shap value
    mean_shap_values_df = all_shap_values_df.groupby('feature_names').mean()

    mean_shap_values_df['count'] = feature_count['mean_shap_values']

    mean_shap_values_df = mean_shap_values_df[mean_shap_values_df['count'] >= df.shape[0] * strict_mean]

    # sort by mean shap value
    mean_shap_values_df.sort_values(by='mean_shap_values', ascending=False, inplace=True)

    return mean_shap_values_df

def get_mean_contribution_general(scores, strict_mean=0.25, adjust_for_accuracy=False, accuracy_scores=None):
    '''
    Sums up the mean score values for each feature, and returns a dataframe with the feature names and the mean score values
    Input:
        scores: iterable with a tuple of (iteration_no, feature_names, scores)
        strict mean: feature must be present in at least x% of iterations
        adjust_for_accuracy: if True, the mean shap value is divided by the accuracy metric of the model
        accuracy scores must be provided if adjust_for_accuracy is True
        accuracy_scores: iterable with a tuple of (feature_names, accuracy_scores)  
    Output: 
        mean_scores_df: dataframe with feature names and mean scores, sorted by mean scores
            col 1: feature_names
            col 2: mean scores
            col 3: count of feature_names
    '''
    
    if hasattr(scores, 'shape'): 
        new_df = scores.copy()
    else: 
        new_df = pd.DataFrame(scores, columns=['iteration_no', 'feature_names', 'scores'])
        
    # get the mean scores for each feature
    mean_scores_df = new_df.groupby('feature_names').mean()

    # mean_scores_df.head()

    if adjust_for_accuracy: 
        
        assert accuracy_scores is not None, 'accuracy_scores must be provided if adjust_for_accuracy is True'
        
        # print(accuracy_scores)
        
        # get the accuracy scores for each feature
        accuracy_scores_df = pd.DataFrame(accuracy_scores, columns=['feature_names', 'accuracy_scores'])
        
        # get the mean accuracy scores for each feature
        mean_accuracy_scores_df = accuracy_scores_df.groupby('feature_names').mean()
        
        # # join the accuracy scores to the mean scores
        mean_scores_df = mean_scores_df.join(mean_accuracy_scores_df, on='feature_names')
        
        # # divide the mean scores by the accuracy scores
        mean_scores_df['scores'] = mean_scores_df['scores'] * mean_scores_df['accuracy_scores']
        
        # # drop the accuracy scores column
        mean_scores_df.drop(columns=['accuracy_scores'], inplace=True)

    # get the count of each feature
    feature_count = new_df.groupby('feature_names').count()

    # join the count to the mean scores
    mean_scores_df['count'] = feature_count['scores']

    # mean_scores_df.head()

    # # filter out features that are not present in at least x% of iterations
    mean_scores_df = mean_scores_df[mean_scores_df['count'] >= new_df['iteration_no'].nunique() * strict_mean]

    # sort by mean scores
    mean_scores_df.sort_values(by='scores', ascending=False, inplace=True)
    
    return mean_scores_df    
    
def get_variation(df, condition, use_iqr=True, strict_mean=0.75, col_name='feature_importance'):
    
    # NOTE: df must contain a column called 'col_name' with a tuple of (features, scores)
    # a column called 'rng' with the rng values is also required
    # ideally, n needs to be large enough to get a good estimate of the variability
    # NOTE: variability_score is the relative standard deviation of the feature importance scores 
    # for mean/std calculations
    
    # first, condition is needed to filter the dataframe
    df = df[df['condition'] == condition]
    
    list_of_dict = []
    feature_importance_df = df[col_name]
    for row in feature_importance_df:
        for x,y in zip(row[0], row[1]):
            # print(f'Feature: {x}, Score: {y}')
            list_of_dict.append({'feature': x, 'variability_score': y})
            
    feature_importance_all = pd.DataFrame(list_of_dict)
    feature_count = feature_importance_all.groupby('feature').count()
    
    if use_iqr: 
        iqr_df = (feature_importance_all.groupby('feature').quantile(0.75) - feature_importance_all.groupby('feature').quantile(0.25))
        iqr_df_div_mean = iqr_df / abs(feature_importance_all.groupby('feature').mean())
        # add IQR and mean columns
        iqr_df_div_mean['iqr'] = iqr_df
        iqr_df_div_mean['mean'] = feature_importance_all.groupby('feature').mean()

        iqr_df_div_mean['count'] = feature_count['variability_score']
        iqr_df_div_mean = iqr_df_div_mean[iqr_df_div_mean['count'] >= df['rng'].nunique() * strict_mean]

        # sort the dataframe by iqr, ascending
        iqr_df_div_mean = iqr_df_div_mean.sort_values(by='variability_score', ascending=True)
        

        return iqr_df_div_mean
    else:
        # calculate the std for each feature divided by the mean
        # NOTE: only use this if a normal distribution is assumed
        std_df = feature_importance_all.groupby('feature').std() / abs(feature_importance_all.groupby('feature').mean())

        # add std and mean columns
        std_df['std'] = feature_importance_all.groupby('feature').std()
        std_df['mean'] = feature_importance_all.groupby('feature').mean()
        
        std_df['count'] = feature_count['variability_score']
        std_df = std_df[std_df['count'] >= df['rng'].nunique() * strict_mean]

        # sort the dataframe by std, ascending

        std_df = std_df.sort_values(by='variability_score', ascending=True)
        return std_df
    
def get_diff_between_feature_contributions(shap_df1: pd.DataFrame, shap_df2: pd.DataFrame):
    # compute the difference in mean shap values for each feature
    diff = shap_df2.copy()
    diff['mean_shap_values'] = shap_df2['mean_shap_values'] - shap_df1['mean_shap_values']
    return diff

def get_abs_sum_for_feature_contributions(df: pd.DataFrame):
    return np.abs(df['mean_shap_values']).sum()

### --- Utils function 

def quick_save_powerkit_results(total_df, meta_df, rngs, condition, file_save_path):
    # save results
    total_df.to_pickle(f'{file_save_path}total_df_{condition}.pkl')
    meta_df.to_pickle(f'{file_save_path}meta_df_{condition}.pkl')
    
    # save rngs
    with open(f'{file_save_path}rngs_list_{condition}.pkl', 'wb') as f:
        pickle.dump(rngs, f)

def build_dataframe(selected_indices, scores, feature_data): 
    # if selected_indices are not indices but labels, skip the label conversion step 
    if not isinstance(selected_indices[0], int) and not isinstance(selected_indices[0], np.int64):
        labels = selected_indices
    else:
        labels = feature_data.columns[selected_indices]
    df = pd.DataFrame({'Selected': selected_indices, 'Scores': scores}, index=labels)
    sorted_df = df.sort_values(by='Scores', ascending=False)
    return sorted_df

def ensemble_k_rank_select(k: int, selection_methods: list, method_kwargs: list, feature_data: pd.DataFrame, label_data):
    all_dfs = []
    assert len(selection_methods) == len(method_kwargs), 'Number of methods and method_kwargs must be equal'
    for idx, method in enumerate(selection_methods):
        selected, score = method(feature_data, label_data, feature_data.shape[0], **method_kwargs[idx])
        df = build_dataframe(selected, score, feature_data)
        all_dfs.append(df)
    
    all_labels = union_df_labels(k,all_dfs)
    return all_labels

def union_df_labels(k, df_list):
    all_labels = []
    for df in df_list:
        k_best_labels = df.index.tolist()[:k]
        all_labels.extend(k_best_labels)
    
    all_labels = list(set(all_labels))
    return all_labels

def get_shuffled_scores(shuffled_label_data, feature_data, selection_method, method_kwargs, verbose=1, n_jobs=1):
    feature_size = feature_data.shape[1]
    if n_jobs == -1:
        n_jobs = cpu_count()
    if n_jobs == 1:
        all_scores = []
        for i, label_data in enumerate(shuffled_label_data):
            selected, score = selection_method(feature_data, label_data, feature_size, **method_kwargs)
            all_scores.extend(score)
            if verbose == 1: 
                print(f'Finished {i+1} out of {len(shuffled_label_data)}')
    else: 
        # use joblib to parallelize the process
        def run_one(label_data):
            selected, score = selection_method(feature_data, label_data, feature_size, **method_kwargs)
            return score
        # process shuffled data 20 at a time, so that we can see progress, then concatenate at the end
        divide_n = 20
        if n_jobs > 20:
            divide_n = 40
        shuffled_label_data_chunks = [shuffled_label_data[i:i + divide_n] for i in range(0, len(shuffled_label_data), divide_n)]
        all_scores = []
        for i, chunk in enumerate(shuffled_label_data_chunks):
            scores = Parallel(n_jobs=n_jobs)(delayed(run_one)(label_data) for label_data in chunk)
            all_scores.extend([score for sublist in scores for score in sublist])
            if verbose == 1: 
                print(f'Finished {i+1} out of {len(shuffled_label_data_chunks)} chunks')
                
    return all_scores