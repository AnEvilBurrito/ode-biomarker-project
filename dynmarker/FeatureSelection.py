# extract joint datasets into feature and labels
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, ElasticNet
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import clone

import numpy as np 

def create_feature_and_label(df: pd.DataFrame, label_name: str = 'LN_IC50'):
    # extract the feature data from the joined dataset

    feature_data = df.drop(columns=[label_name])
    
    # if the column 'CELLLINE' is present, drop it
    if 'CELLLINE' in feature_data.columns:
        feature_data.drop(columns=['CELLLINE'], inplace=True)

    # extract the label data from the joined dataset

    label_data = df[label_name]

    return feature_data, label_data

def naive_test_regression(feature_data, label_data, cv=None, verbose=0):

    model_list = [LinearRegression(), LinearSVR(max_iter=15000), KNeighborsRegressor(), RandomForestRegressor(), MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=2000)]
    model_names = ['Linear Regression', 'Linear SVR', 'KNN', 'Random Forest', 'MLP']
    scores = []

    if cv is None:
        # use simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.2, random_state=42)
        for model, name in zip(model_list, model_names):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
            if verbose == 1:
                print(f'------ Naive test - {name}')
                print(f'{name} score: {score:.4f}')
            scores.append(score)

    else:
        for model, name in zip(model_list, model_names):
            score = cross_val_score(model, feature_data, label_data, cv=cv, scoring='neg_mean_squared_error')
            mean, std = -score.mean(), score.std()
            scores.append((mean, std))
            if verbose == 1:
                print(f'------ Naive test - {name}')
                print(f'{name} score: {mean:.4f}, std: {std:.4f}')
    
    return list(zip(model_names, scores))


def naive_test_classification(feature_data, label_data, cv=None):

    model_list = [LogisticRegression(), LinearSVC(max_iter=15000), KNeighborsClassifier(), RandomForestClassifier(), MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000)]
    model_names = ['Logistic Regression', 'Linear SVC', 'KNN', 'Random Forest', 'MLP']

    if cv is None:
        # use simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.2, random_state=42)
        for model, name in zip(model_list, model_names):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f'------ Naive test - {name}')
            print(f'{name} score: {accuracy_score(y_test, y_pred):.4f}')

    else:
        for model, name in zip(model_list, model_names):
            score = cross_val_score(model, feature_data, label_data, cv=cv, scoring='accuracy')
            print(f'------ Naive test - {name}')
            print(f'{name} score: {score.mean():.4f}, std: {score.std():.4f}')


def filter_feature_selection(feature_data, label_data, method, K, get_selected_data=False):
    # define feature selection
    fs = SelectKBest(score_func=method, k=K)
    # apply feature selection
    pearson_fs_feature = fs.fit_transform(feature_data, label_data)
    print(pearson_fs_feature.shape)

    scores, pval = fs.scores_, fs.pvalues_
    support = fs.get_support(indices=True)

    selected_scores, selected_pval = scores[support], pval[support]

    # rank selected features by highest score

    ranked_features = sorted(zip(selected_scores, support), reverse=True)

    # selected features 
    if get_selected_data:
        selected_feature_data = feature_data.iloc[:, support]
        return selected_feature_data, support
    
    # print(ranked_features)
    return selected_scores, support

def mrmr_select_fcq(X, y, K, verbose=0, return_index=True):

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

def greedy_forward_select(X, y, K, model, ranked_features, scoring_method, cv_num=5, verbose=0):
    # ------------ Input
    # WARNING: This function is not properly validated. Its results are not reliable. 
    # K[Int]: number of features to be selected
    # X[pd.Dataframe]: feature data
    # y[pd.Series]: label data
    # model: model to be used for feature selection
    # ranked_features[List[Tuple]]: list of tuples of (feature score, feature index)
    # scoring_method[String]: scoring method for cross validation
    # verbose[Int] = 0: reporting information for each iteration


    # ------------ Output
    # feature_selected[List[Int]]: list of selected features index format 
    # successive_scores[List[Float]]: list of successive scores

    feature_selected = []
    # select the first feature
    feature_selected.append(ranked_features[0][1])

    selected_feature_data_gffs = X.iloc[:,feature_selected]
    score = cross_val_score(model, selected_feature_data_gffs, y, cv=cv_num, scoring=scoring_method).mean()
    successive_scores = []
    successive_scores.append(score)

    while len(feature_selected) < K:
        max_score = -10000
        max_feature = 0
        for i in range(len(ranked_features)):
            if ranked_features[i][1] not in feature_selected:
                feature_selected.append(ranked_features[i][1])
                selected_feature_data_gffs = X.iloc[:,feature_selected]

                # cross validation 5 times and get average score
                score = cross_val_score(model, selected_feature_data_gffs, y, cv=cv_num, scoring=scoring_method).mean()
                if score > max_score:
                    max_score = score
                    max_feature = ranked_features[i][1]
                feature_selected.pop()
        feature_selected.append(max_feature)
        successive_scores.append(max_score)
        if verbose == 1:
            print(f'Feature Selected: {max_feature}, Score: {max_score}, Feature Size: {len(feature_selected)}')

    return feature_selected, successive_scores

def greedy_forward_select_evaluate(feature_data, label_data, K, model, ranked_features, scoring_method, outer_cv, inner_cv, verbose=0):

    # Nested CV with greedy forward selection
    feature_selected_per_fold = []
    final_score = []
    for i, (train_index, test_index) in enumerate(outer_cv.split(feature_data, label_data)):
        X_train, X_test = feature_data.iloc[train_index,:], feature_data.iloc[test_index,:]
        y_train, y_test = label_data.iloc[train_index], label_data.iloc[test_index]
        print(f'------------ Outer CV: {i+1}')
        feature_selected = []
        # select the first feature
        feature_selected.append(ranked_features[0][1])

        selected_feature_data_gffs = X_train.iloc[:,feature_selected]
        score = cross_val_score(model, selected_feature_data_gffs, y_train, cv=inner_cv, scoring=scoring_method).mean()
        successive_scores = []
        successive_scores.append(score)

        while len(feature_selected) < K:
            max_score = -10000
            max_feature = 0
            for i in range(len(ranked_features)):
                if ranked_features[i][1] not in feature_selected:
                    feature_selected.append(ranked_features[i][1])
                    selected_feature_data_gffs = X_train.iloc[:,feature_selected]

                    # cross validation 5 times and get average score
                    score = cross_val_score(model, selected_feature_data_gffs, y_train, cv=inner_cv, scoring=scoring_method).mean()
                    if score > max_score:
                        max_score = score
                        max_feature = ranked_features[i][1]
                    feature_selected.pop()
            feature_selected.append(max_feature)
            successive_scores.append(max_score)
            if verbose == 1:
                print(f'Feature Selected: {max_feature}, Score: {max_score}, Feature Size: {len(feature_selected)}')

        # Fit the model with selected features, on all x_train and y_train
        model.fit(X_train.iloc[:,feature_selected], y_train)
        # Evaluate the model on all x_test and y_test
        y_pred = model.predict(X_test.iloc[:,feature_selected])
        score = mean_squared_error(y_test, y_pred)
        final_score.append(score)

        print(f'------------ Final Score: {score}')

    print(f'Final Score: {np.mean(final_score)}')

    return feature_selected_per_fold, final_score

def grand_random_selection(feature_data, k):

    # get indices of features
    feature_indices = np.arange(feature_data.shape[1])
    # randomly shuffle indices
    np.random.shuffle(feature_indices)
    # select k indices
    selected_indices = feature_indices[:k]
    # select features based on indices
    selected_features = feature_data.iloc[:,selected_indices]

    return selected_features, selected_indices

def comparative_random_feature_selection(feature_data, selected_features_indices):

    # get indices of features
    feature_indices = np.arange(feature_data.shape[1])

    # get a new index, which is not in the selected_features_indices
    new_index = np.random.choice(np.setdiff1d(feature_indices, selected_features_indices))

    # append new index to selected_features_indices
    new_selected_features_indices = np.append(selected_features_indices, new_index)

    # select features based on indices
    new_selected_features = feature_data.iloc[:,new_selected_features_indices]

    return new_selected_features, new_selected_features_indices

from sklearn.base import BaseEstimator, TransformerMixin


class MRMRFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.feature_names]
    

def mrmr_select_fcq_sklearn(X, y, K, verbose=0):

    # ------------ Input
    # X: pandas.DataFrame, features
    # y: pandas.Series, target variable
    # K: number of features to select
    
    # ------------ Output
    # scores: np.array, score for each feature at their respective index
    #          depends on the K parameter, other features automatically get a score of 0

    # if X is a numpy array, convert to pandas DataFrame
    if type(X) == np.ndarray:
        X = pd.DataFrame(X)
    
    # if y is a numpy array, convert to pandas Series
    if type(y) == np.ndarray:
        y = pd.Series(y)

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
    

    # create ranking

    scores = np.zeros(X.shape[1])
    scores[[X.columns.get_loc(c) for c in selected]] = successive_scores

    return scores


def example_run_model_func(X, y, i, k, model, n_fold_splits, verbose=0, **kwargs):
    '''
    parameters
        X: pandas.DataFrame, features
        y: pandas.Series, target variable
        i: int, outer fold number
        k: int, number of features to select
        model: sklearn model
        n_fold_splits: int, number of inner folds
        verbose: int, 0 or 1
        kwargs: dict, optional arguments
        keepModel: bool, if True, keep the model and return it, else return None
    returns 
       all_dfs: list of pandas.DataFrame, each element is a dataframe of the results of each inner fold
    '''

    all_dfs = []
    keepModel = kwargs.get('keepModel', False)
    outer_cv = KFold(n_splits=n_fold_splits, shuffle=True)
    scores = []
    for j, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # print(f'------------ CV Numver: {i}')   
        
        random_feature_train, random_indices = grand_random_selection(X.iloc[train_index,:], k=k)

        # print(f'{i}, {j}, {k}, {model.__class__.__name__}, {random_indices}')

        model = clone(model)

        model.fit(X_train.iloc[:, random_indices], y_train)

        y_pred = model.predict(X_test.iloc[:, random_indices])
        mse = mean_squared_error(y_test, y_pred)
        scores.append(mse)
        # print(f'MSE: {mse:.4f}')

        if keepModel == True:
            new_df = pd.DataFrame({'i': [i], 'model_name': [model.__class__.__name__], 'k': [k], 'model': [model], 'cv_number': [j], 'feature_indices': [random_indices], 'eval_score': [mse]})
        else:
            new_df = pd.DataFrame({'i': [i], 'model_name': [model.__class__.__name__], 'k': [k], 'model': [None], 'cv_number': [j], 'feature_indices': [random_indices], 'eval_score': [mse]})

        all_dfs.append(new_df)
        # add the evaluation instance to the dataframe
        # evaluation_df = pd.concat((evaluation_df, new_df), ignore_index=True)

    if verbose == 1:
        print(f'-- Iteration: {i}, feature size: {k}, model: {model.__class__.__name__}, Avg Eval Score: {np.mean(scores):.4f}, Std Eval Score: {np.std(scores):.4f}')
    
    return all_dfs