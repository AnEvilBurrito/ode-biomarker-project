from sklearn.base import clone 
from joblib import Parallel, delayed, cpu_count
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class EvaluationPipeline:

    def __init__(self):
        '''
        This class is used to run a function in parallel or serial and store the results in a pandas dataframe
        Generic Pipeline that allows for customisation of dataframes
        Falls back to pipeline by sklearn if function fails
        '''

        self.evaluation_df = pd.DataFrame(columns=['i', 'model_name', 'k', 'model', 'cv_number', 'feature_indices', 'eval_score'])

        self.run_model_func = None
        self.function_set = False

    def set_function(self, func):
        self.run_model_func = func 
        self.function_set = True

    def run_function(self, X, y, iterations, k_ranges, model_list, n_fold_splits=5, n_cores=1, **kwargs):

        # clean the evaluation dataframe
        self.evaluation_df = pd.DataFrame(columns=['i', 'model_name', 'k', 'model', 'cv_number', 'feature_indices', 'eval_score'])

        if self.function_set == False:
            raise ValueError('EvaluationPipeline: The function to be run has not been set')
        if n_cores == 1:
            # run the function in serial, single core, allows for printing
            for model in model_list:
                for i in range(iterations):
                    for k in k_ranges:
                        df_cv = self.run_model_func(X, y, i, k, model, n_fold_splits, verbose=1, **kwargs)
                        for df in df_cv:
                            self.evaluation_df = pd.concat((self.evaluation_df, df), ignore_index=True)
                print(f'--- Finished {model.__class__.__name__} using {n_cores} core ---')
        elif n_cores == -1:
            # run the function in parallel with all available cores
            models = model_list
            eval_list_total = []
            for model in models:
                evaluation_list = Parallel(n_jobs=cpu_count())(delayed(self.run_model_func)(X, y, i, k, model, n_fold_splits, verbose=0, **kwargs) for i in range(iterations) for k in k_ranges)
                for evaluation in evaluation_list:
                    for ev in evaluation:
                        eval_list_total.append(ev)
                print(f'--- Finished {model.__class__.__name__} using {cpu_count()} cores ---')
            self.evaluation_df = pd.concat(eval_list_total, ignore_index=True)
        else:
            # run the function in parallel with n_jobs cores
            models = model_list
            eval_list_total = []
            for model in models:
                evaluation_list = Parallel(n_jobs=n_cores)(delayed(self.run_model_func)(X, y, i, k, model, n_fold_splits, verbose=0, **kwargs) for i in range(iterations) for k in k_ranges)
                for evaluation in evaluation_list:
                    for ev in evaluation:
                        eval_list_total.append(ev)
                print(f'--- Finished {model.__class__.__name__} using {n_cores} cores ---')
            self.evaluation_df = pd.concat(eval_list_total, ignore_index=True)
