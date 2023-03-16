from sklearn.base import clone 
from joblib import Parallel, delayed, cpu_count
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

import itertools

class GeneralPipeline:

    def __init__(self, df_columns):
        '''
        This class is used to run a function in parallel or serial and store the results in a pandas dataframe
        Generic Pipeline that allows for customisation of dataframes
        Falls back to pipeline by sklearn if function fails
        '''

        self.evaluation_df = pd.DataFrame(columns=df_columns)
        self.run_model_func = None
        self.function_set = False

        self.plotting_func = None
        self.plotting_set = False

    def set_function(self, func):
        self.run_model_func = func 
        self.function_set = True

    def set_plotting_function(self, func):
        self.plotting_func = func
        self.plotting_set = True

    def run_function(self, model_list, iterations, n_cores=1, verbose=0, **kwargs):

        # clean the evaluation dataframe
        self.evaluation_df = pd.DataFrame(columns=self.evaluation_df.columns)

        all_ranges = [item for item in kwargs.values() if type(item) == list or type(item) == np.ndarray]
        all_ranges.insert(0, range(iterations))

        if self.function_set == False:
            raise ValueError('EvaluationPipeline: The function to be run has not been set')
        if n_cores == 1:
            # run the function in serial, single core, allows for printing
            for model in model_list:
                for xs in itertools.product(*all_ranges):
                    df = self.run_model_func(model, verbose=verbose, *xs, **kwargs)
                    pd.concat((self.evaluation_df, df), ignore_index=True)
                print(f'--- Finished {model.__class__.__name__} using {n_cores} core ---')
        else:
            # run the function in parallel with n_jobs cores
            if n_cores > cpu_count():
                raise ValueError(f'EvaluationPipeline: The number of cores requested ({n_cores}) is greater than the number of cores available ({cpu_count()})')
            if n_cores == -1:
                n_cores = cpu_count()

            models = model_list
            eval_list_total = []
            for model in models:
                evaluation_list = Parallel(n_jobs=n_cores)(delayed(self.run_model_func)(model, verbose=0, *xs, **kwargs) for xs in itertools.product(*all_ranges))
                for evaluation in evaluation_list:
                    eval_list_total.append(evaluation)
                print(f'--- Finished {model.__class__.__name__} using {n_cores} cores ---')
            self.evaluation_df = pd.concat(eval_list_total, ignore_index=True)

    def run_plot(self):
        if self.plotting_set == False:
            raise ValueError('EvaluationPipeline: The plotting function has not been set')
        self.plotting_func(self.evaluation_df)