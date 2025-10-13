# Jupyter notebook in Jupytext format

# %% [markdown]
# ## Initialisation

# %%
import os

# get current working directory
path = os.getcwd()

# find the string 'project' in the path, return index (fallback to cwd if not found)
index_project = path.find('project')
project_path = path[: index_project + 7] if index_project != -1 else path

# set the working directory to project root (as in the notebook)
os.chdir(project_path)
print(f'Project path set to: {os.getcwd()}')

from PathLoader import PathLoader  # noqa: E402
from DataLink import DataLink  # noqa: E402

# environment filenames used in the notebook
path_loader = PathLoader('data_config.env', 'current_user.env')
data_link = DataLink(path_loader, 'data_codes.csv')

# output folder and experiment id (same as notebook)
folder_name = "ThesisResult3-BenchmarkingFeatureSelection"
exp_id = "v1"

# prepare results directory and file save path
save_dir = os.path.join(path_loader.get_data_path(), 'data', 'results', folder_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
file_save_path = f"{save_dir}{os.sep}"

# Example dataset loading codes from the notebook (attempted but protected with try/except)
loading_code = "goncalves-gdsc-2-Palbociclib-LN_IC50-sin"
try:
    proteomic_feature_data, proteomic_label_data = data_link.get_data_using_code(loading_code)
    print(f'Proteomic feature data shape: {proteomic_feature_data.shape}', f'Proteomic label data shape: {proteomic_label_data.shape}')
except Exception as e:
    print(f'Could not load proteomic data for code {loading_code}: {e}')

loading_code = "ccle-gdsc-2-Palbociclib-LN_IC50"
try:
    ccle_feature_data, ccle_label_data = data_link.get_data_using_code(loading_code)
    print(f'CCLE feature data shape: {ccle_feature_data.shape}', f'CCLE label data shape: {ccle_label_data.shape}')
except Exception as e:
    print(f'Could not load CCLE data for code {loading_code}: {e}')

# %%

# %% [markdown]
# ## Feature Selection Benchmarking Functions

# %%
import numpy as np
import pandas as pd
from typing import Dict, List, Literal
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

from toolkit import (
    FirstQuantileImputer, 
    f_regression_select, 
    get_model_from_string,
    mrmr_select_fcq,
    greedy_feedforward_select,
    greedy_feedforward_select_sy,  # noqa: E402
    relieff_select,
    select_network_features
)

# %% [markdown]
# ### Core Feature Selection Pipeline

# %%
def _drop_correlated_columns(X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """Drop highly correlated columns (memory-efficient version)."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in sorted(upper.columns):
        if col in to_drop:
            continue
        high_corr = upper.index[upper[col] > threshold].tolist()
        to_drop.update(high_corr)
    return [c for c in X.columns if c not in to_drop]

def feature_selection_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rng: int,
    *,
    k: int = 100,
    var_threshold: float = 0.0,
    corr_threshold: float = 0.95,
    model_name: Literal[
        "LinearRegression",
        "RandomForestRegressor",
        "SVR",
    ] = "RandomForestRegressor",
    selection_method: Literal[
        "f_regression", "mrmr", "gffs", "relieff", "network"
    ] = "f_regression",
    network_data: Dict = None,  # For network-based selection
    max_network_distance: int = 2,  # For network-based selection
) -> Dict:
    """
    Generic pipeline for benchmarking different feature selection methods.
    """
    # 0) Sanitize inputs
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    y_train = pd.Series(y_train).replace([np.inf, -np.inf], np.nan)
    mask = ~y_train.isna()
    X_train, y_train = X_train.loc[mask], y_train.loc[mask]

    # 1) Impute on train
    imputer = FirstQuantileImputer().fit(X_train)
    Xtr = imputer.transform(X_train, return_df=True).astype(float)
    Xtr = Xtr.fillna(0.0)

    # 2) Variance filter
    n_features_initial = Xtr.shape[1]
    vt = VarianceThreshold(threshold=var_threshold).fit(Xtr)
    vt_keep_cols = Xtr.columns[vt.get_support()].tolist()
    Xtr = Xtr[vt_keep_cols]
    n_features_post_variance = Xtr.shape[1]

    # 3) Correlation filter
    corr_keep_cols = _drop_correlated_columns(Xtr, threshold=corr_threshold)
    Xtr = Xtr[corr_keep_cols]
    n_features_post_correlation = Xtr.shape[1]

    # 4) Feature selection based on method
    k_sel = min(k, Xtr.shape[1]) if Xtr.shape[1] > 0 else 0
    
    if k_sel == 0:
        selected_features, selector_scores = [], np.array([])
        sel_train = Xtr.iloc[:, :0]
        no_features = True
    else:
        if selection_method == "f_regression":
            selected_features, selector_scores = f_regression_select(Xtr, y_train, k=k_sel)
        
        elif selection_method == "mrmr":
            selected_indices, selector_scores = mrmr_select_fcq(Xtr, y_train, K=k_sel, verbose=0)
            selected_features = Xtr.columns[selected_indices].tolist()
        
        elif selection_method == "gffs":
            # Greedy Feedforward Selection with RandomForest
            model = RandomForestRegressor(n_estimators=50, random_state=rng)
            start_feature = Xtr.columns[np.argmax(np.abs(Xtr.corrwith(y_train)))]
            selected_features, selector_scores, _ = greedy_feedforward_select_sy(
                Xtr, y_train, k_sel, model, start_feature, cv=3, verbose=0
            )
        
        elif selection_method == "relieff":
            selected_indices, selector_scores = relieff_select(Xtr, y_train, k=k_sel, n_jobs=1)
            selected_features = Xtr.columns[selected_indices].tolist()
        
        elif selection_method == "network" and network_data is not None:
            # Network-based feature selection
            selected_features, sel_train_network = select_network_features(
                Xtr, y_train, network_data, max_network_distance
            )
            # Apply statistical filter on network features
            if len(selected_features) > k_sel:
                selected_features, selector_scores = f_regression_select(
                    Xtr[selected_features], y_train, k=k_sel
                )
            else:
                selector_scores = np.ones(len(selected_features))
        
        else:
            # Fallback to f_regression
            selected_features, selector_scores = f_regression_select(Xtr, y_train, k=k_sel)
        
        sel_train = Xtr[selected_features]
        no_features = False

    # 5) Fixed model; robust fallback if no features
    if no_features:
        model = DummyRegressor(strategy="mean")
        model.fit(np.zeros((len(y_train), 1)), y_train)
        model_type = "DummyRegressor(mean)"
        model_params = {"strategy": "mean"}
    else:
        if model_name == "LinearRegression":
            model = get_model_from_string("LinearRegression")
        elif model_name == "RandomForestRegressor":
            model = get_model_from_string(
                "RandomForestRegressor", n_estimators=100, random_state=rng
            )
        elif model_name == "SVR":
            model = get_model_from_string("SVR", kernel="linear", C=1.0)
        else:
            raise ValueError("Unsupported model_name for feature selection benchmarking.")
        
        model.fit(sel_train, y_train)
        model_type = model_name
        try:
            model_params = model.get_params(deep=False)
        except Exception:
            model_params = {}

    return {
        "imputer": imputer,
        "vt_keep_cols": vt_keep_cols,
        "corr_keep_cols": corr_keep_cols,
        "selected_features": list(selected_features),
        "selector_scores": np.array(selector_scores),
        "k_requested": int(k),
        "k_effective": int(len(selected_features)),
        "n_features_initial": int(n_features_initial),
        "n_features_post_variance": int(n_features_post_variance),
        "n_features_post_correlation": int(n_features_post_correlation),
        "var_threshold": float(var_threshold),
        "corr_threshold": float(corr_threshold),
        "selection_method": selection_method,
        "model": model,
        "model_type": model_type,
        "model_params": model_params,
        "train_data": sel_train,
        "rng": int(rng),
        "no_features": bool(no_features),
        "n_train_samples_used": int(len(y_train)),
    }

# %%

# %% [markdown]
# ### Evaluation Function

# %%
def feature_selection_eval(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    pipeline_components: Dict,
    metric_primary: Literal["r2", "pearson_r", "spearman_r"] = "r2",
    importance_from: Literal["selector", "model"] = "selector",
) -> Dict:
    """
    Evaluation function for feature selection benchmarking.
    """
    # Unpack pipeline components
    imputer = pipeline_components["imputer"]
    vt_keep = set(pipeline_components["vt_keep_cols"])
    corr_keep = set(pipeline_components["corr_keep_cols"])
    selected = list(pipeline_components["selected_features"])
    selector_scores = pipeline_components["selector_scores"]
    model = pipeline_components["model"]
    model_name = pipeline_components["model_type"]
    selection_method = pipeline_components.get("selection_method", "f_regression")

    # 0) Sanitize test inputs
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    y_test = pd.Series(y_test).replace([np.inf, -np.inf], np.nan)
    mask_y = ~y_test.isna()
    X_test, y_test = X_test.loc[mask_y], y_test.loc[mask_y]

    # Apply identical transforms
    Xti = imputer.transform(X_test, return_df=True).astype(float).fillna(0.0)
    cols_after_vt = [c for c in Xti.columns if c in vt_keep]
    Xti = Xti[cols_after_vt]
    cols_after_corr = [c for c in Xti.columns if c in corr_keep]
    Xti = Xti[cols_after_corr]
    Xsel = Xti[selected] if len(selected) > 0 else Xti.iloc[:, :0]

    # Predict robustly
    if pipeline_components.get("no_features", False) or Xsel.shape[1] == 0:
        y_pred = np.full_like(
            y_test.values, fill_value=float(y_test.mean()), dtype=float
        )
    else:
        y_pred = np.asarray(model.predict(Xsel), dtype=float)

    # Filter any non-finite values before metrics
    mask_fin = np.isfinite(y_test.values) & np.isfinite(y_pred)
    y_t = y_test.values[mask_fin]
    y_p = y_pred[mask_fin]
    n_test_used = int(y_t.shape[0])

    if n_test_used < 2:
        r2 = np.nan
        pearson_r = pearson_p = np.nan
        spearman_rho = spearman_p = np.nan
    else:
        r2 = r2_score(y_t, y_p)
        pearson_r, pearson_p = pearsonr(y_t, y_p)
        spearman_rho, spearman_p = spearmanr(y_t, y_p)

    metrics = {
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else np.nan,
        "pearson_p": float(pearson_p) if np.isfinite(pearson_p) else np.nan,
        "spearman_rho": float(spearman_rho) if np.isfinite(spearman_rho) else np.nan,
        "spearman_p": float(spearman_p) if np.isfinite(spearman_p) else np.nan,
        "n_test_samples_used": n_test_used,
    }

    # Feature importance
    if importance_from == "selector":
        fi = (np.array(selected), np.array(selector_scores))
    else:
        if hasattr(model, "feature_importances_") and len(selected) > 0:
            fi = (np.array(selected), model.feature_importances_)
        else:
            fi = (np.array(selected), np.zeros(len(selected)))

    primary = metrics.get(metric_primary, metrics["r2"])

    return {
        "feature_importance": fi,
        "feature_importance_from": importance_from,
        "model_performance": float(primary) if primary is not None else np.nan,
        "metrics": metrics,
        "selection_method": selection_method,
        "model_name": model_name,
        "selected_features": selected,
        "selector_scores": np.array(selector_scores),
        "y_pred": y_p,
        "y_true_index": y_test.index[mask_fin],
    }

# %%

# %% [markdown]
# ### Experiment Setup Functions

# %%
def setup_feature_selection_experiments(
    feature_data: pd.DataFrame,
    label_data: pd.Series,
    network_data: Dict = None
) -> Dict:
    """
    Set up Powerkit instances for feature selection benchmarking.
    Returns a dictionary with Powerkit instances for different experimental setups.
    """
    from toolkit import Powerkit
    
    # Create Powerkit instance
    pk = Powerkit(feature_data, label_data)
    
    # Define selection methods to test (Phase 1: Individual methods)
    selection_methods = ["f_regression", "mrmr", "gffs", "relieff"]
    if network_data is not None:
        selection_methods.append("network")
    
    # Define k values to test
    k_values = [20, 50, 100, 200]
    
    # Fixed parameters
    fixed_args = {
        "var_threshold": 0.0,
        "corr_threshold": 0.95,
        "model_name": "RandomForestRegressor"
    }
    
    # Phase 1: Individual feature selection methods (3.3.3.1)
    for method in selection_methods:
        for k in k_values:
            condition_name = f"{method}_k{k}"
            
            # Special handling for network method
            pipeline_args = {**fixed_args, "k": k, "selection_method": method}
            if method == "network" and network_data is not None:
                pipeline_args.update({
                    "network_data": network_data,
                    "max_network_distance": 2
                })
            
            pk.add_condition(
                condition=condition_name,
                get_importance=True,
                pipeline_function=feature_selection_pipeline,
                pipeline_args=pipeline_args,
                eval_function=feature_selection_eval,
                eval_args={"metric_primary": "r2", "importance_from": "selector"},
            )
    
    # Phase 2: Hybrid selection strategies (3.3.3.4)
    if network_data is not None:
        hybrid_methods = ["union", "intersection", "prioritization"]
        best_stat_method = "f_regression"  # Will be determined from Phase 1 results
        
        for hybrid_method in hybrid_methods:
            for k in k_values:
                condition_name = f"hybrid_{hybrid_method}_k{k}"
                
                # Create custom pipeline for hybrid methods
                def hybrid_pipeline(X_train, y_train, rng, **kwargs):
                    # Use the hybrid selection function
                    if hybrid_method == "union":
                        selected_features, selector_scores = hybrid_union_select(
                            X_train, y_train, k, best_stat_method, "network", network_data
                        )
                    elif hybrid_method == "intersection":
                        selected_features, selector_scores = hybrid_intersection_select(
                            X_train, y_train, k, best_stat_method, "network", network_data
                        )
                    elif hybrid_method == "prioritization":
                        selected_features, selector_scores = hybrid_prioritization_select(
                            X_train, y_train, k, best_stat_method, "network", network_data
                        )
                    else:
                        # Fallback to union
                        selected_features, selector_scores = hybrid_union_select(
                            X_train, y_train, k, best_stat_method, "network", network_data
                        )
                    
                    # Apply standard preprocessing
                    imputer = FirstQuantileImputer().fit(X_train)
                    Xtr = imputer.transform(X_train, return_df=True).astype(float).fillna(0.0)
                    
                    # Variance and correlation filtering
                    vt = VarianceThreshold(threshold=0.0).fit(Xtr)
                    vt_keep_cols = Xtr.columns[vt.get_support()].tolist()
                    Xtr = Xtr[vt_keep_cols]
                    
                    corr_keep_cols = _drop_correlated_columns(Xtr, threshold=0.95)
                    Xtr = Xtr[corr_keep_cols]
                    
                    # Select only the hybrid-selected features
                    sel_train = Xtr[selected_features]
                    
                    # Train model
                    model = get_model_from_string("RandomForestRegressor", n_estimators=100, random_state=rng)
                    model.fit(sel_train, y_train)
                    
                    return {
                        "imputer": imputer,
                        "vt_keep_cols": vt_keep_cols,
                        "corr_keep_cols": corr_keep_cols,
                        "selected_features": selected_features,
                        "selector_scores": selector_scores,
                        "k_requested": k,
                        "k_effective": len(selected_features),
                        "model": model,
                        "model_type": "RandomForestRegressor",
                        "train_data": sel_train,
                        "rng": rng,
                        "n_train_samples_used": len(y_train),
                    }
                
                pk.add_condition(
                    condition=condition_name,
                    get_importance=True,
                    pipeline_function=hybrid_pipeline,
                    pipeline_args={},
                    eval_function=feature_selection_eval,
                    eval_args={"metric_primary": "r2", "importance_from": "selector"},
                )
    
    return {"powerkit": pk, "selection_methods": selection_methods, "k_values": k_values}

# %%

# %% [markdown]
# ### Data Preparation

# %%
def prepare_feature_selection_data():
    """
    Prepare data for feature selection benchmarking.
    Aligns indices and ensures data consistency.
    """
    # Use proteomic data (best performing from 3.3.2)
    loading_code = "goncalves-gdsc-2-Palbociclib-LN_IC50-sin"
    try:
        feature_data, label_data = data_link.get_data_using_code(loading_code)
        print(f'Feature data shape: {feature_data.shape}, Label data shape: {label_data.shape}')
        
        # Ensure numeric only
        feature_data = feature_data.select_dtypes(include=[np.number])
        
        return feature_data, label_data
        
    except Exception as e:
        print(f'Error loading data: {e}')
        return None, None

# %%

# %% [markdown]
# ### Network Interaction Distance Function

# %%
def network_interaction_distance_select(
    X: pd.DataFrame, 
    y: pd.Series, 
    k: int, 
    network_data: Dict,
    max_distance: int = 2,
    **kwargs
):
    """
    Feature selection based on network interaction distance.
    Selects features that are within max_distance of target genes in the network.
    """
    # Get target genes (features most correlated with y)
    correlations = X.corrwith(y).abs()
    target_genes = correlations.nlargest(10).index.tolist()  # Top 10 correlated genes
    
    # Get network neighbors within max_distance
    network_features = set()
    for target in target_genes:
        for distance in range(1, max_distance + 1):
            if distance in network_data and target in network_data[distance]:
                network_features.update(network_data[distance][target])
    
    # Convert to list and filter to features present in X
    network_features = [f for f in network_features if f in X.columns]
    
    if len(network_features) == 0:
        # Fallback to statistical selection if no network features found
        return f_regression_select(X, y, k)
    
    # Apply statistical filter on network features
    if len(network_features) > k:
        selected_features, scores = f_regression_select(
            X[network_features], y, k
        )
    else:
        selected_features = network_features
        scores = np.ones(len(network_features))
    
    return selected_features, scores

# %%

# %% [markdown]
# ### Hybrid Selection Strategies

# %%
def hybrid_union_select(
    X: pd.DataFrame, 
    y: pd.Series, 
    k: int, 
    method1: str = "f_regression",
    method2: str = "network",
    network_data: Dict = None,
    **kwargs
):
    """
    Hybrid selection: union of features from two methods.
    """
    # Get features from first method
    if method1 == "f_regression":
        features1, scores1 = f_regression_select(X, y, k)
    elif method1 == "mrmr":
        indices1, scores1 = mrmr_select_fcq(X, y, k)
        features1 = X.columns[indices1].tolist()
    elif method1 == "network" and network_data is not None:
        features1, scores1 = network_interaction_distance_select(X, y, k, network_data)
    else:
        features1, scores1 = f_regression_select(X, y, k)
    
    # Get features from second method
    if method2 == "f_regression":
        features2, scores2 = f_regression_select(X, y, k)
    elif method2 == "mrmr":
        indices2, scores2 = mrmr_select_fcq(X, y, k)
        features2 = X.columns[indices2].tolist()
    elif method2 == "network" and network_data is not None:
        features2, scores2 = network_interaction_distance_select(X, y, k, network_data)
    else:
        features2, scores2 = f_regression_select(X, y, k)
    
    # Union of features
    union_features = list(set(features1 + features2))
    
    if len(union_features) > k:
        # Re-rank union features using f_regression
        selected_features, scores = f_regression_select(X[union_features], y, k)
    else:
        selected_features = union_features
        scores = np.ones(len(union_features))
    
    return selected_features, scores

def hybrid_intersection_select(
    X: pd.DataFrame, 
    y: pd.Series, 
    k: int, 
    method1: str = "f_regression",
    method2: str = "network",
    network_data: Dict = None,
    **kwargs
):
    """
    Hybrid selection: intersection of features from two methods.
    """
    # Get features from first method
    if method1 == "f_regression":
        features1, scores1 = f_regression_select(X, y, k)
    elif method1 == "mrmr":
        indices1, scores1 = mrmr_select_fcq(X, y, k)
        features1 = X.columns[indices1].tolist()
    elif method1 == "network" and network_data is not None:
        features1, scores1 = network_interaction_distance_select(X, y, k, network_data)
    else:
        features1, scores1 = f_regression_select(X, y, k)
    
    # Get features from second method
    if method2 == "f_regression":
        features2, scores2 = f_regression_select(X, y, k)
    elif method2 == "mrmr":
        indices2, scores2 = mrmr_select_fcq(X, y, k)
        features2 = X.columns[indices2].tolist()
    elif method2 == "network" and network_data is not None:
        features2, scores2 = network_interaction_distance_select(X, y, k, network_data)
    else:
        features2, scores2 = f_regression_select(X, y, k)
    
    # Intersection of features
    intersection_features = list(set(features1) & set(features2))
    
    if len(intersection_features) > k:
        # Re-rank intersection features using f_regression
        selected_features, scores = f_regression_select(X[intersection_features], y, k)
    else:
        selected_features = intersection_features
        scores = np.ones(len(intersection_features))
    
    return selected_features, scores

def hybrid_prioritization_select(
    X: pd.DataFrame, 
    y: pd.Series, 
    k: int, 
    method1: str = "f_regression",
    method2: str = "network",
    network_data: Dict = None,
    **kwargs
):
    """
    Hybrid selection: prioritize features based on combined scores.
    """
    # Get features and scores from both methods
    if method1 == "f_regression":
        features1, scores1 = f_regression_select(X, y, X.shape[1])  # Get all features
    elif method1 == "mrmr":
        indices1, scores1 = mrmr_select_fcq(X, y, X.shape[1])
        features1 = X.columns[indices1].tolist()
    elif method1 == "network" and network_data is not None:
        features1, scores1 = network_interaction_distance_select(X, y, X.shape[1], network_data)
    else:
        features1, scores1 = f_regression_select(X, y, X.shape[1])
    
    if method2 == "f_regression":
        features2, scores2 = f_regression_select(X, y, X.shape[1])
    elif method2 == "mrmr":
        indices2, scores2 = mrmr_select_fcq(X, y, X.shape[1])
        features2 = X.columns[indices2].tolist()
    elif method2 == "network" and network_data is not None:
        features2, scores2 = network_interaction_distance_select(X, y, X.shape[1], network_data)
    else:
        features2, scores2 = f_regression_select(X, y, X.shape[1])
    
    # Create score dictionaries
    score_dict1 = dict(zip(features1, scores1))
    score_dict2 = dict(zip(features2, scores2))
    
    # Combine scores (multiplicative combination)
    combined_scores = {}
    all_features = set(features1 + features2)
    
    for feature in all_features:
        score1 = score_dict1.get(feature, 0)
        score2 = score_dict2.get(feature, 0)
        combined_scores[feature] = score1 * score2
    
    # Select top k features based on combined scores
    sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    selected_features = [f[0] for f in sorted_features[:k]]
    scores = [f[1] for f in sorted_features[:k]]
    
    return selected_features, scores

# %%

# %% [markdown]
# ### Network Data Loading

# %%
def load_network_data():
    """
    Load network interaction data for network-based feature selection.
    """
    try:
        # Load STRING network data
        network_file_path = f'{path_loader.get_data_path()}data/protein-interaction/STRING/palbociclib_nth_degree_neighbours.pkl'
        if os.path.exists(network_file_path):
            import pickle
            with open(network_file_path, 'rb') as f:
                nth_degree_neighbours = pickle.load(f)
            print(f"Loaded network data with {len(nth_degree_neighbours)} distance levels")
            return nth_degree_neighbours
        else:
            print("Network data file not found")
            return None
    except Exception as e:
        print(f"Error loading network data: {e}")
        return None

# %%

# %% [markdown]
# ### Consensus and Stability Analysis

# %%
def analyze_feature_stability(df_results: pd.DataFrame, conditions: List[str] = None):
    """
    Analyze feature selection stability across different conditions and CV folds.
    """
    if conditions is None:
        conditions = df_results['condition'].unique()
    
    stability_results = {}
    
    for condition in conditions:
        condition_data = df_results[df_results['condition'] == condition]
        
        # Collect all selected features across iterations
        all_selected_features = []
        for _, row in condition_data.iterrows():
            selected_features = row['selected_features']
            if isinstance(selected_features, list):
                all_selected_features.extend(selected_features)
        
        # Calculate selection frequency
        feature_counts = pd.Series(all_selected_features).value_counts()
        total_iterations = len(condition_data)
        selection_frequency = feature_counts / total_iterations
        
        # Calculate Jaccard similarity between iterations
        jaccard_scores = []
        feature_sets = []
        for _, row in condition_data.iterrows():
            selected_features = row['selected_features']
            if isinstance(selected_features, list):
                feature_sets.append(set(selected_features))
        
        # Calculate pairwise Jaccard similarities
        for i in range(len(feature_sets)):
            for j in range(i + 1, len(feature_sets)):
                set1 = feature_sets[i]
                set2 = feature_sets[j]
                if len(set1) > 0 and len(set2) > 0:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    jaccard_scores.append(jaccard)
        
        stability_results[condition] = {
            'selection_frequency': selection_frequency,
            'mean_jaccard_similarity': np.mean(jaccard_scores) if jaccard_scores else 0,
            'std_jaccard_similarity': np.std(jaccard_scores) if jaccard_scores else 0,
            'total_iterations': total_iterations,
            'unique_features_selected': len(feature_counts),
            'top_features': selection_frequency.head(10).to_dict()
        }
    
    return stability_results

def compare_method_stability(stability_results: Dict, method_groups: Dict):
    """
    Compare stability between different groups of methods.
    """
    comparison_results = {}
    
    for group_name, methods in method_groups.items():
        group_stability = []
        for method in methods:
            if method in stability_results:
                group_stability.append({
                    'method': method,
                    'mean_jaccard': stability_results[method]['mean_jaccard_similarity'],
                    'unique_features': stability_results[method]['unique_features_selected'],
                    'top_features': stability_results[method]['top_features']
                })
        
        if group_stability:
            comparison_results[group_name] = {
                'methods': group_stability,
                'mean_jaccard_across_methods': np.mean([m['mean_jaccard'] for m in group_stability]),
                'mean_unique_features': np.mean([m['unique_features'] for m in group_stability])
            }
    
    return comparison_results

def create_stability_visualizations(stability_results: Dict, save_path: str = None):
    """
    Create visualizations for feature selection stability analysis.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data for plotting
    methods = list(stability_results.keys())
    jaccard_scores = [stability_results[m]['mean_jaccard_similarity'] for m in methods]
    unique_features = [stability_results[m]['unique_features_selected'] for m in methods]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Jaccard similarity
    sns.barplot(x=methods, y=jaccard_scores, ax=ax1)
    ax1.set_title('Mean Jaccard Similarity by Method')
    ax1.set_ylabel('Jaccard Similarity')
    ax1.set_xlabel('Feature Selection Method')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Unique features selected
    sns.barplot(x=methods, y=unique_features, ax=ax2)
    ax2.set_title('Unique Features Selected by Method')
    ax2.set_ylabel('Number of Unique Features')
    ax2.set_xlabel('Feature Selection Method')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}stability_analysis.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig

# %%

# %% [markdown]
# ### Experiment Execution Functions

# %%
def run_feature_selection_benchmarking(
    feature_data: pd.DataFrame,
    label_data: pd.Series,
    network_data: Dict = None,
    n_iterations: int = 100,
    n_jobs: int = -1
) -> Dict:
    """
    Execute the complete feature selection benchmarking pipeline.
    """
    # Set up experiments
    experiment_setup = setup_feature_selection_experiments(feature_data, label_data, network_data)
    pk = experiment_setup['powerkit']
    
    # Generate RNGs for repeated holdout
    rngs = np.random.RandomState(42).randint(0, 100000, size=n_iterations)
    
    # Run all conditions
    print("Running feature selection benchmarking experiments...")
    df_results = pk.run_all_conditions(rng_list=rngs, n_jobs=n_jobs, verbose=True)
    
    # Add metadata
    df_results['experiment_type'] = 'feature_selection_benchmarking'
    df_results['n_iterations'] = n_iterations
    
    # Analyze stability
    print("Analyzing feature selection stability...")
    stability_results = analyze_feature_stability(df_results)
    
    # Compare method groups
    method_groups = {
        'statistical': ['f_regression_k20', 'f_regression_k50', 'f_regression_k100', 'f_regression_k200'],
        'wrapper': ['gffs_k20', 'gffs_k50', 'gffs_k100', 'gffs_k200'],
        'filter': ['mrmr_k20', 'mrmr_k50', 'mrmr_k100', 'mrmr_k200'],
        'network': ['network_k20', 'network_k50', 'network_k100', 'network_k200'],
        'hybrid': ['hybrid_union_k20', 'hybrid_union_k50', 'hybrid_union_k100', 'hybrid_union_k200',
                  'hybrid_intersection_k20', 'hybrid_intersection_k50', 'hybrid_intersection_k100', 'hybrid_intersection_k200',
                  'hybrid_prioritization_k20', 'hybrid_prioritization_k50', 'hybrid_prioritization_k100', 'hybrid_prioritization_k200']
    }
    
    comparison_results = compare_method_stability(stability_results, method_groups)
    
    return {
        'df_results': df_results,
        'stability_results': stability_results,
        'comparison_results': comparison_results,
        'experiment_setup': experiment_setup
    }

def save_benchmarking_results(results: Dict, save_path: str):
    """
    Save benchmarking results to files.
    """
    import pickle
    
    # Save main results dataframe
    results['df_results'].to_pickle(f"{save_path}feature_selection_results.pkl")
    
    # Save stability analysis
    with open(f"{save_path}stability_results.pkl", 'wb') as f:
        pickle.dump(results['stability_results'], f)
    
    # Save comparison results
    with open(f"{save_path}comparison_results.pkl", 'wb') as f:
        pickle.dump(results['comparison_results'], f)
    
    print(f"Results saved to {save_path}")

# %%

# %% [markdown]
# ## Complete Experiment Execution

# %%
def execute_complete_benchmarking():
    """
    Execute the complete feature selection benchmarking pipeline.
    """
    print("=== FEATURE SELECTION BENCHMARKING PIPELINE ===")
    
    # 1. Load network data
    print("1. Loading network data...")
    network_data = load_network_data()
    
    # 2. Prepare data
    print("2. Preparing feature selection data...")
    feature_data, label_data = prepare_feature_selection_data()
    
    if feature_data is None or label_data is None:
        print("Error: Could not load feature selection data")
        return None
    
    # 3. Run benchmarking
    print("3. Running feature selection benchmarking...")
    results = run_feature_selection_benchmarking(
        feature_data, label_data, network_data, n_iterations=50, n_jobs=-1
    )
    
    # 4. Save results
    print("4. Saving results...")
    save_benchmarking_results(results, file_save_path)
    
    # 5. Create visualizations
    print("5. Creating visualizations...")
    create_stability_visualizations(results['stability_results'], file_save_path)
    
    print("=== BENCHMARKING COMPLETED ===")
    return results

# %%

# %% [markdown]
# ## Next Steps

# %%
# This cell will be used for executing the experiments in subsequent steps
print("Feature selection benchmarking functions implemented successfully.")
print("To execute the complete pipeline, run:")
print("results = execute_complete_benchmarking()")
print("")
print("This will execute all phases of the benchmarking:")
print("1. Phase 1: Individual method benchmarking (3.3.3.1)")
print("2. Phase 2: Feature size optimization (3.3.3.2)") 
print("3. Phase 3: Network method benchmarking (3.3.3.3)")
print("4. Phase 4: Hybrid strategies (3.3.3.4)")
print("5. Phase 5: Consensus and stability analysis (3.3.3.5)")

# %%
