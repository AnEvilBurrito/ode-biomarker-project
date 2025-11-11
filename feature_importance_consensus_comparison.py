#!/usr/bin/env python3
"""
Extended script for comparing different feature selection approaches using Random Forest model
Compares 4 methods: clinical+network, network-only, MRMR-only, and network+MRMR
Uses k=500 features, network distance 3, and split size 0.3
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Literal
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
import shap  # For SHAP feature importance

# Import custom modules
from PathLoader import PathLoader
from DataLink import DataLink
from toolkit import (
    FirstQuantileImputer, 
    f_regression_select, 
    get_model_from_string,
    mrmr_select_fcq_fast,
    select_network_features,
    select_random_features,
    Powerkit
)

# Import protein ID mapper
from protein_id_mapper import load_protein_mapping, map_network_features, filter_available_features


def extract_uniprot_id(feature_name):
    """
    Extract Uniprot ID from proteomics feature name format: [Gene][UniprotID]:HUMAN
    Example: 'EGFRP00533:HUMAN' -> 'P00533'
    """
    import re
    match = re.search(r'([A-Z][0-9]{5})', feature_name)
    if match:
        return match.group(1)
    return None


def get_clinical_features(X: pd.DataFrame, clinical_genes_df: pd.DataFrame) -> List[str]:
    """
    Extract clinical features from proteomics data based on clinical gene mapping
    """
    # Extract Uniprot IDs from proteomics feature names
    feature_uniprot_ids = {}
    for col in X.columns:
        uniprot_id = extract_uniprot_id(col)
        if uniprot_id:
            feature_uniprot_ids[col] = uniprot_id

    # Get clinical gene Uniprot IDs
    clinical_uniprot_ids = set(clinical_genes_df['UniProt ID'].tolist())

    # Find overlapping Uniprot IDs
    overlapping_ids = clinical_uniprot_ids.intersection(set(feature_uniprot_ids.values()))

    # Create list of clinical features
    clinical_feature_columns = []
    for clinical_id in overlapping_ids:
        matching_features = [feature for feature, uniprot_id in feature_uniprot_ids.items() 
                            if uniprot_id == clinical_id]
        clinical_feature_columns.extend(matching_features)

    return clinical_feature_columns


def clinical_network_select_wrapper(X: pd.DataFrame, y: pd.Series, k: int, 
                                   nth_degree_neighbours: list, max_distance: int,
                                   clinical_genes_df: pd.DataFrame) -> tuple:
    """
    Clinical + Network feature selection wrapper
    Combines clinical features with network features (no MRMR filtering, no k truncation)
    """
    # Step 1: Get clinical features
    clinical_features = get_clinical_features(X, clinical_genes_df)
    print(f"Found {len(clinical_features)} clinical features")
    
    # Step 2: Get network features within specified distance
    if max_distance <= len(nth_degree_neighbours):
        network_features = nth_degree_neighbours[max_distance - 1] if nth_degree_neighbours[max_distance - 1] is not None else []
    else:
        network_features = []
    
    # Map network features to proteomics-compatible IDs
    path_loader = PathLoader("data_config.env", "current_user.env")
    mapping_df = load_protein_mapping(path_loader)
    
    if mapping_df is not None:
        mapped_network_features = map_network_features(network_features, mapping_df)
        available_network_features = filter_available_features(mapped_network_features, X.columns)
        network_features = available_network_features
    else:
        available_network_features = [f for f in network_features if f in X.columns]
        network_features = available_network_features
    
    print(f"Found {len(network_features)} network features (distance {max_distance})")
    
    # Step 3: Combine clinical and network features (remove duplicates)
    combined_features = list(set(clinical_features + network_features))
    print(f"Combined clinical+network features: {len(combined_features)}")
    
    # If no features available, return empty selection
    if len(combined_features) == 0:
        return [], np.array([])
    
    # Return all combined features (no MRMR filtering, no k truncation)
    selected_features = combined_features
    
    # Create dummy scores (all 1.0 since no ranking is performed)
    scores = np.ones(len(selected_features))
    
    return selected_features, scores


def network_only_select_wrapper(X: pd.DataFrame, y: pd.Series, k: int, 
                               nth_degree_neighbours: list, max_distance: int) -> tuple:
    """
    Network-only feature selection wrapper
    Selects features from network distance 3 only (no MRMR filtering, no k truncation)
    """
    # Step 1: Get network features within specified distance
    if max_distance <= len(nth_degree_neighbours):
        network_features = nth_degree_neighbours[max_distance - 1] if nth_degree_neighbours[max_distance - 1] is not None else []
    else:
        network_features = []
    
    # Map network features to proteomics-compatible IDs
    path_loader = PathLoader("data_config.env", "current_user.env")
    mapping_df = load_protein_mapping(path_loader)
    
    if mapping_df is not None:
        mapped_network_features = map_network_features(network_features, mapping_df)
        available_network_features = filter_available_features(mapped_network_features, X.columns)
        network_features = available_network_features
    else:
        available_network_features = [f for f in network_features if f in X.columns]
        network_features = available_network_features
    
    print(f"Network-only features (distance {max_distance}): {len(network_features)}")
    
    # If no network features available, return empty selection
    if len(network_features) == 0:
        return [], np.array([])
    
    # Return all network features (no MRMR filtering, no k truncation)
    selected_features = network_features
    
    # Create dummy scores (all 1.0 since no ranking is performed)
    scores = np.ones(len(selected_features))
    
    return selected_features, scores


def mrmr_only_select_wrapper(X: pd.DataFrame, y: pd.Series, k: int) -> tuple:
    """
    MRMR-only feature selection wrapper
    Applies MRMR to all available features
    """
    print(f"MRMR-only selection from {X.shape[1]} features")
    
    # Apply MRMR to all features
    k_actual = min(k, X.shape[1])
    selected_features, scores = mrmr_select_fcq_fast(X, y, k_actual, verbose=0)
    
    return selected_features, scores


def mrmr_network_select_wrapper(X: pd.DataFrame, y: pd.Series, k: int, 
                               nth_degree_neighbours: list, max_distance: int) -> tuple:
    """
    Joint MRMR + Network feature selection wrapper (original method)
    First selects network features, then applies MRMR to select k from network subset
    """
    # Step 1: Get network features within specified distance
    if max_distance <= len(nth_degree_neighbours):
        network_features = nth_degree_neighbours[max_distance - 1] if nth_degree_neighbours[max_distance - 1] is not None else []
    else:
        network_features = []
    
    # If no network features available, return empty selection
    if len(network_features) == 0:
        return [], np.array([])
    
    # Step 1.5: Map network protein IDs to proteomics-compatible IDs
    path_loader = PathLoader("data_config.env", "current_user.env")
    mapping_df = load_protein_mapping(path_loader)
    
    if mapping_df is not None:
        mapped_network_features = map_network_features(network_features, mapping_df)
        available_network_features = filter_available_features(mapped_network_features, X.columns)
        network_features = available_network_features
    else:
        available_network_features = [f for f in network_features if f in X.columns]
        network_features = available_network_features
    
    # If no mapped network features available, return empty selection
    if len(network_features) == 0:
        return [], np.array([])
    
    # Step 2: Apply MRMR to select k features from network subset
    k_actual = min(k, len(network_features))
    
    # Get the network subset of features
    X_network = X[network_features]
    
    # Apply MRMR to network subset
    selected_features, scores = mrmr_select_fcq_fast(X_network, y, k_actual, verbose=0)
    
    return selected_features, scores


def _drop_correlated_columns(X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """Drop highly correlated columns to reduce redundancy"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in sorted(upper.columns):
            if col in to_drop:
                continue
            high_corr = upper.index[upper[col] > threshold].tolist()
            to_drop.update(high_corr)
        return [c for c in X.columns if c not in to_drop]


def create_feature_importance_pipeline(
    selection_method: callable, k: int, method_name: str, model_name: str,
    importance_method: Literal["shap", "mdi"] = "mdi"
):
    """
    Create pipeline for model selection with feature importance calculation
    Supports both SHAP and MDI (Mean Decrease Impurity) importance methods
    """

    def pipeline_function(X_train: pd.DataFrame, y_train: pd.Series, rng: int):
        # 1) Sanitize inputs and imputation
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        y_train = pd.Series(y_train).replace([np.inf, -np.inf], np.nan)
        mask = ~y_train.isna()
        X_train, y_train = X_train.loc[mask], y_train.loc[mask]

        # 2) Imputation
        imputer = FirstQuantileImputer().fit(X_train)
        Xtr = imputer.transform(X_train, return_df=True).astype(float)
        Xtr = Xtr.fillna(0.0)

        # 3) Correlation filtering (applied to both train and test)
        corr_keep_cols = _drop_correlated_columns(Xtr, threshold=0.95)
        Xtr_filtered = Xtr[corr_keep_cols]

        # 4) Feature selection with timing measurement
        k_sel = min(k, Xtr_filtered.shape[1]) if Xtr_filtered.shape[1] > 0 else 0
        if k_sel == 0:
            selected_features, selector_scores = [], np.array([])
            no_features = True
            feature_selection_time = 0.0
        else:
            # Measure feature selection time
            feature_selection_start = time.time()
            selected_features, selector_scores = selection_method(
                Xtr_filtered, y_train, k_sel
            )
            feature_selection_time = time.time() - feature_selection_start
            no_features = False

        # 5) Standardization and model training with timing
        scaler = None  # Initialize scaler to None
        
        if no_features or len(selected_features) == 0:
            model = DummyRegressor(strategy="mean")
            model_type = "DummyRegressor(mean)"
            model_params = {"strategy": "mean"}
            sel_train = Xtr_filtered.iloc[:, :0]
            model_training_time = 0.0
        else:
            sel_train = Xtr_filtered[selected_features]
            scaler = StandardScaler()
            sel_train_scaled = scaler.fit_transform(sel_train)
            sel_train_scaled = pd.DataFrame(
                sel_train_scaled, index=sel_train.index, columns=selected_features
            )

            # Train model with timing
            model_training_start = time.time()
            
            # Use standard hyperparameters for RandomForestRegressor
            if model_name == "RandomForestRegressor":
                model = get_model_from_string("RandomForestRegressor", n_estimators=100, random_state=rng)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            model.fit(sel_train_scaled, y_train)
            model_training_time = time.time() - model_training_start
            model_type = model_name
            model_params = (
                model.get_params(deep=False) if hasattr(model, "get_params") else {}
            )

        return {
            "imputer": imputer,
            "corr_keep_cols": corr_keep_cols,
            "selected_features": list(selected_features),
            "selector_scores": np.array(selector_scores),
            "model": model,
            "model_type": model_type,
            "model_params": model_params,
            "scaler": scaler,
            "no_features": no_features,
            "rng": rng,
            "feature_selection_time": feature_selection_time,
            "model_training_time": model_training_time,
            "importance_method": importance_method,  # Store which importance method to use
        }

    return pipeline_function


def feature_importance_eval(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    pipeline_components: Dict,
    metric_primary: Literal["r2", "pearson_r", "spearman_r"] = "r2",
) -> Dict:
    """
    Evaluation function for feature importance analysis
    Supports both SHAP and MDI feature importance methods
    """

    # Unpack components
    imputer = pipeline_components["imputer"]
    corr_keep = set(pipeline_components["corr_keep_cols"])
    selected = list(pipeline_components["selected_features"])
    selector_scores = pipeline_components["selector_scores"]
    model = pipeline_components["model"]
    model_name = pipeline_components["model_type"]
    scaler = pipeline_components.get("scaler", None)
    no_features = pipeline_components.get("no_features", False)
    feature_selection_time = pipeline_components.get("feature_selection_time", 0.0)
    model_training_time = pipeline_components.get("model_training_time", 0.0)
    importance_method = pipeline_components.get("importance_method", "mdi")

    # Apply identical transforms as training
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    y_test = pd.Series(y_test).replace([np.inf, -np.inf], np.nan)
    mask_y = ~y_test.isna()
    X_test, y_test = X_test.loc[mask_y], y_test.loc[mask_y]

    Xti = imputer.transform(X_test, return_df=True).astype(float).fillna(0.0)

    # Apply same correlation filtering as training
    cols_after_corr = [c for c in Xti.columns if c in corr_keep]
    Xti = Xti[cols_after_corr]

    # Select features
    Xsel = Xti[selected] if len(selected) > 0 else Xti.iloc[:, :0]

    # Standardize if scaler exists (i.e., features were selected)
    if scaler is not None and len(selected) > 0:
        Xsel_scaled = scaler.transform(Xsel)
        Xsel_scaled = pd.DataFrame(Xsel_scaled, index=Xsel.index, columns=selected)
    else:
        Xsel_scaled = Xsel

    # Predict with timing
    prediction_start = time.time()
    if no_features or Xsel.shape[1] == 0:
        y_pred = np.full_like(
            y_test.values, fill_value=float(y_test.mean()), dtype=float
        )
    else:
        y_pred = np.asarray(model.predict(Xsel_scaled), dtype=float)
    prediction_time = time.time() - prediction_start

    # Calculate metrics
    mask_fin = np.isfinite(y_test.values) & np.isfinite(y_pred)
    y_t = y_test.values[mask_fin]
    y_p = y_pred[mask_fin]

    if len(y_t) < 2:
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
        "n_test_samples_used": len(y_t),
    }

    # Feature importance calculation
    if not no_features and len(selected) > 0:
        if importance_method == "shap":
            # Calculate SHAP values
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Xsel_scaled)
                # For regression, shap_values might be 2D, take mean values
                if len(shap_values.shape) == 2:
                    feature_importance_abs = np.abs(shap_values).mean(axis=0)  # Absolute values
                    feature_importance_signed = shap_values.mean(axis=0)       # Original signed values
                else:
                    feature_importance_abs = np.abs(shap_values)
                    feature_importance_signed = shap_values
                
                fi = (np.array(selected), feature_importance_abs)  # Backward compatible: absolute values
                fi_signed = (np.array(selected), feature_importance_signed)  # New: signed values
                importance_source = "shap"
            except Exception as e:
                print(f"SHAP calculation failed: {e}, falling back to MDI")
                # Fall back to MDI if SHAP fails
                if hasattr(model, "feature_importances_"):
                    fi = (np.array(selected), model.feature_importances_)
                    fi_signed = None  # MDI doesn't have signed values
                else:
                    fi = (np.array(selected), np.zeros(len(selected)))
                    fi_signed = None
                importance_source = "mdi_fallback"
        
        elif importance_method == "mdi":
            # Use MDI (Mean Decrease Impurity)
            if hasattr(model, "feature_importances_"):
                fi = (np.array(selected), model.feature_importances_)
                fi_signed = None  # MDI doesn't have signed values
            elif model_name in ("LinearRegression",) and len(selected) > 0:
                coef = getattr(model, "coef_", np.zeros(len(selected)))
                fi = (np.array(selected), np.abs(coef))
                fi_signed = None
            else:
                fi = (np.array(selected), np.zeros(len(selected)))
                fi_signed = None
            importance_source = "mdi"
    else:
        fi = (np.array(selected), np.zeros(len(selected)))
        fi_signed = None
        importance_source = "none"

    primary = metrics.get(metric_primary, metrics["r2"])

    return {
        "feature_importance": fi,  # Backward compatible: absolute values
        "feature_importance_signed": fi_signed,  # New: signed values (SHAP only, MDI=None)
        "feature_importance_from": importance_source,
        "model_performance": float(primary) if primary is not None else np.nan,
        "metrics": metrics,
        "selected_features": selected,
        "model_name": model_name,
        "selected_scores": selector_scores,
        "k": len(selected),
        "rng": pipeline_components.get("rng", None),
        "y_pred": y_p,
        "y_true_index": y_test.index[mask_fin],
        "feature_selection_time": feature_selection_time,
        "model_training_time": model_training_time,
        "prediction_time": prediction_time,
        "importance_method": importance_method,
    }


class PowerkitWithVariableSplit(Powerkit):
    """Extended Powerkit class that allows variable split sizes"""
    
    def __init__(self, feature_data: pd.DataFrame, label_data: pd.Series, cv_split_size: float = 0.1) -> None:
        super().__init__(feature_data, label_data)
        self.cv_split_size = cv_split_size
    
    def set_split_size(self, split_size: float):
        """Set the cross-validation split size"""
        self.cv_split_size = split_size


def save_consensus_feature_importance(total_df, condition, file_save_path):
    """
    Save consensus feature importance results to dedicated files
    Now supports both absolute and signed importance values
    """
    # Filter results for the specific condition
    condition_df = total_df[total_df['condition'] == condition]
    
    # Extract absolute feature importance data (backward compatible)
    all_importance_data = []
    for _, row in condition_df.iterrows():
        feature_names = row['feature_importance'][0]
        importance_scores = row['feature_importance'][1]
        rng = row['rng']
        
        for feature_name, score in zip(feature_names, importance_scores):
            all_importance_data.append({
                'iteration_rng': rng,
                'feature_name': feature_name,
                'importance_score': score,
                'importance_method': row.get('importance_method', 'unknown')
            })
    
    # Create iteration-level importance DataFrame for absolute values
    iteration_importance_df = pd.DataFrame(all_importance_data)
    
    # Calculate consensus importance (mean across iterations) for absolute values
    consensus_importance = iteration_importance_df.groupby('feature_name').agg({
        'importance_score': ['mean', 'std', 'count'],
        'importance_method': 'first'
    }).round(6)
    
    # Flatten column names
    consensus_importance.columns = ['_'.join(col).strip() for col in consensus_importance.columns.values]
    consensus_importance = consensus_importance.rename(columns={
        'importance_score_mean': 'mean_importance',
        'importance_score_std': 'std_importance',
        'importance_score_count': 'occurrence_count',
        'importance_method_first': 'importance_method'
    })
    
    # Sort by mean importance
    consensus_importance = consensus_importance.sort_values('mean_importance', ascending=False)
    
    # Save consensus results for absolute values
    consensus_file = f"{file_save_path}consensus_feature_importance_{condition}.pkl"
    consensus_importance.to_pickle(consensus_file)
    
    # Save iteration-level data for absolute values
    iteration_file = f"{file_save_path}iteration_feature_importance_{condition}.pkl"
    iteration_importance_df.to_pickle(iteration_file)
    
    # Extract signed feature importance data if available (SHAP only)
    all_signed_importance_data = []
    for _, row in condition_df.iterrows():
        if row.get('feature_importance_signed') is not None:
            feature_names = row['feature_importance_signed'][0]
            signed_scores = row['feature_importance_signed'][1]
            rng = row['rng']
            
            for feature_name, score in zip(feature_names, signed_scores):
                all_signed_importance_data.append({
                    'iteration_rng': rng,
                    'feature_name': feature_name,
                    'importance_score_signed': score,
                    'importance_method': row.get('importance_method', 'unknown')
                })
    
    # Create iteration-level importance DataFrame for signed values (if available)
    if all_signed_importance_data:
        iteration_signed_importance_df = pd.DataFrame(all_signed_importance_data)
        
        # Calculate consensus importance for signed values
        consensus_signed_importance = iteration_signed_importance_df.groupby('feature_name').agg({
            'importance_score_signed': ['mean', 'std', 'count'],
            'importance_method': 'first'
        }).round(6)
        
        # Flatten column names
        consensus_signed_importance.columns = ['_'.join(col).strip() for col in consensus_signed_importance.columns.values]
        consensus_signed_importance = consensus_signed_importance.rename(columns={
            'importance_score_signed_mean': 'mean_importance_signed',
            'importance_score_signed_std': 'std_importance_signed',
            'importance_score_signed_count': 'occurrence_count',
            'importance_method_first': 'importance_method'
        })
        
        # Sort by mean signed importance
        consensus_signed_importance = consensus_signed_importance.sort_values('mean_importance_signed', ascending=False)
        
        # Save consensus results for signed values
        consensus_signed_file = f"{file_save_path}consensus_feature_importance_signed_{condition}.pkl"
        consensus_signed_importance.to_pickle(consensus_signed_file)
        
        # Save iteration-level data for signed values
        iteration_signed_file = f"{file_save_path}iteration_feature_importance_signed_{condition}.pkl"
        iteration_signed_importance_df.to_pickle(iteration_signed_file)
        
        return consensus_importance, iteration_importance_df, consensus_signed_importance, iteration_signed_importance_df
    
    return consensus_importance, iteration_importance_df, None, None


def compare_feature_selection_methods(all_consensus_results, file_save_path, condition_base):
    """
    Compare feature importance rankings across different selection methods
    """
    # Create comparison DataFrame
    comparison_data = []
    
    # Get all features across all methods
    all_features = set()
    for method_name, consensus_df in all_consensus_results.items():
        all_features.update(consensus_df.index)
    
    # Compare each feature across methods
    for feature in all_features:
        feature_data = {'feature_name': feature}
        
        for method_name, consensus_df in all_consensus_results.items():
            if feature in consensus_df.index:
                row = consensus_df.loc[feature]
                feature_data[f'{method_name}_mean_importance'] = row['mean_importance']
                feature_data[f'{method_name}_std_importance'] = row['std_importance']
                feature_data[f'{method_name}_occurrence_count'] = row['occurrence_count']
            else:
                feature_data[f'{method_name}_mean_importance'] = 0.0
                feature_data[f'{method_name}_std_importance'] = 0.0
                feature_data[f'{method_name}_occurrence_count'] = 0
        
        comparison_data.append(feature_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate method agreement scores
    method_names = list(all_consensus_results.keys())
    for i, method1 in enumerate(method_names):
        for method2 in method_names[i+1:]:
            comparison_df[f'agreement_{method1}_{method2}'] = (
                comparison_df[f'{method1}_mean_importance'] * comparison_df[f'{method2}_mean_importance']
            )
    
    # Sort by average importance across methods
    avg_importance_cols = [f'{method}_mean_importance' for method in method_names]
    comparison_df['avg_importance'] = comparison_df[avg_importance_cols].mean(axis=1)
    comparison_df = comparison_df.sort_values('avg_importance', ascending=False)
    
    # Save comparison results
    comparison_file = f"{file_save_path}feature_selection_method_comparison_{condition_base}.pkl"
    comparison_df.to_pickle(comparison_file)
    
    return comparison_df


def analyze_feature_set_overlap(all_consensus_results, file_save_path, condition_base):
    """
    Analyze overlapping features between different selection methods
    """
    method_names = list(all_consensus_results.keys())
    
    # Create overlap analysis
    overlap_data = []
    
    # Calculate pairwise overlaps
    for i, method1 in enumerate(method_names):
        features1 = set(all_consensus_results[method1].index)
        
        for method2 in method_names[i+1:]:
            features2 = set(all_consensus_results[method2].index)
            
            # Calculate overlap metrics
            intersection = features1 & features2
            union = features1 | features2
            jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0
            
            overlap_data.append({
                'method_pair': f"{method1}_vs_{method2}",
                'method1_features': len(features1),
                'method2_features': len(features2),
                'overlap_count': len(intersection),
                'jaccard_similarity': jaccard_similarity,
                'overlap_percentage_method1': len(intersection) / len(features1) if len(features1) > 0 else 0,
                'overlap_percentage_method2': len(intersection) / len(features2) if len(features2) > 0 else 0
            })
    
    # Calculate overall overlap across all methods
    all_features_sets = [set(all_consensus_results[method].index) for method in method_names]
    common_features_all = set.intersection(*all_features_sets) if all_features_sets else set()
    union_all = set.union(*all_features_sets) if all_features_sets else set()
    
    overlap_data.append({
        'method_pair': 'all_methods',
        'method1_features': len(union_all),
        'method2_features': len(common_features_all),
        'overlap_count': len(common_features_all),
        'jaccard_similarity': len(common_features_all) / len(union_all) if len(union_all) > 0 else 0,
        'overlap_percentage_method1': len(common_features_all) / len(union_all) if len(union_all) > 0 else 0,
        'overlap_percentage_method2': 1.0  # All methods share these features
    })
    
    overlap_df = pd.DataFrame(overlap_data)
    
    # Save overlap analysis
    overlap_file = f"{file_save_path}feature_set_overlap_analysis_{condition_base}.pkl"
    overlap_df.to_pickle(overlap_file)
    
    return overlap_df


def main():
    """Main execution function for feature selection method comparison"""
    # Initialize path loader and data link
    path_loader = PathLoader("data_config.env", "current_user.env")
    data_link = DataLink(path_loader, "data_codes.csv")
    
    # Setup experiment parameters
    folder_name = "ThesisResult-FeatureSelectionComparison"
    exp_id = "v1_rf_k500_4methods_split0.3_comparison"
    
    # Create results directory
    if not os.path.exists(f"{path_loader.get_data_path()}data/results/{folder_name}"):
        os.makedirs(f"{path_loader.get_data_path()}data/results/{folder_name}")
    
    file_save_path = f"{path_loader.get_data_path()}data/results/{folder_name}/"
    
    # Create report file
    report_file = f"{file_save_path}feature_selection_comparison_report_{exp_id}.md"
    
    def print_and_save(message, file_handle):
        print(message)
        file_handle.write(message + "\n")
    
    # Start total batch timing
    total_batch_start = time.time()
    
    with open(report_file, 'w', encoding='utf-8') as report:
        print_and_save("# Feature Selection Method Comparison Report", report)
        print_and_save(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}", report)
        print_and_save(f"**Experiment ID:** {exp_id}", report)
        print_and_save("", report)
        
        print_and_save("## Execution Summary", report)
        print_and_save("Comparing 4 feature selection methods using SHAP and MDI importance with consensus-based stability analysis...", report)
        print_and_save(f"Results will be saved to: {file_save_path}", report)
    
    with open(report_file, 'a', encoding='utf-8') as report:
        print_and_save("## Data Loading and Preparation", report)
        print_and_save("Loading proteomics data...", report)
        loading_code = "goncalves-gdsc-2-Palbociclib-LN_IC50-sin"
        proteomic_feature_data, proteomic_label_data = data_link.get_data_using_code(loading_code)
        
        print_and_save(f"Proteomic feature data shape: {proteomic_feature_data.shape}", report)
        print_and_save(f"Proteomic label data shape: {proteomic_label_data.shape}", report)
        
        print_and_save("Preparing and aligning data...", report)
        proteomic_feature_data = proteomic_feature_data.select_dtypes(include=[np.number])
        
        # Align indices
        common_indices = sorted(
            set(proteomic_feature_data.index) & set(proteomic_label_data.index)
        )
        feature_data = proteomic_feature_data.loc[common_indices]
        label_data = proteomic_label_data.loc[common_indices]
        
        print_and_save(f"Final aligned dataset shape: {feature_data.shape}", report)
        print_and_save(f"Final aligned label shape: {label_data.shape}", report)
        
        print_and_save("## Clinical Gene Mapping Loading", report)
        # Load clinical gene mapping
        clinical_genes_df = pd.read_excel("gene_to_uniprot_mapping.xlsx", sheet_name="Sheet1")
        print_and_save(f"Loaded clinical gene mapping with {len(clinical_genes_df)} genes", report)
        
        print_and_save("## Network Structure Loading", report)
        # Load network structure
        network_file_path = f"{path_loader.get_data_path()}data/protein-interaction/STRING/palbociclib_nth_degree_neighbours.pkl"
        print_and_save(f"Loading network structure from: {network_file_path}", report)
        
        with open(network_file_path, 'rb') as f:
            nth_degree_neighbours = pickle.load(f)
        
        print_and_save(f"Network structure loaded. Available distances: {list(range(1, len(nth_degree_neighbours) + 1))}", report)
        # Print the size of each distance
        for distance in range(1, len(nth_degree_neighbours) + 1):
            if distance <= len(nth_degree_neighbours):
                feature_count = len(nth_degree_neighbours[distance - 1]) if nth_degree_neighbours[distance - 1] is not None else 0
                print_and_save(f"Distance {distance}: {feature_count} features", report)
        
        print_and_save("## Experiment Setup", report)
        # Setup experiment parameters
        k_value = 500  # Use k=500 as requested
        split_size = 0.3  # Use split size 0.3 as requested
        model_name = "RandomForestRegressor"
        network_distance = 3  # Use distance 3 as requested
        
        # Define the 4 feature selection methods
        methods = {
            "clinical_network_d3": lambda X, y, k: clinical_network_select_wrapper(
                X, y, k, nth_degree_neighbours, network_distance, clinical_genes_df
            ),
            "network_only_d3": lambda X, y, k: network_only_select_wrapper(
                X, y, k, nth_degree_neighbours, network_distance
            ),
            "mrmr_only": lambda X, y, k: mrmr_only_select_wrapper(X, y, k),
            "mrmr_network_d3": lambda X, y, k: mrmr_network_select_wrapper(
                X, y, k, nth_degree_neighbours, network_distance
            )
        }
        
        # Define conditions for all methods and importance methods
        conditions = []
        for method_name, method_func in methods.items():
            conditions.append(f"{model_name}_k{k_value}_{method_name}_split{split_size}_shap")
            conditions.append(f"{model_name}_k{k_value}_{method_name}_split{split_size}_mdi")
        
        print_and_save(f"Benchmarking {len(conditions)} conditions (4 methods × 2 importance methods)", report)
        print_and_save("Feature Selection Methods:", report)
        print_and_save("- Clinical + Network (distance 3)", report)
        print_and_save("- Network only (distance 3)", report)
        print_and_save("- MRMR only", report)
        print_and_save("- MRMR + Network (distance 3) - original method", report)
        print_and_save(f"Model: {model_name}", report)
        print_and_save(f"Feature Set Size: k={k_value}", report)
        print_and_save(f"Split Size: {split_size}", report)
        print_and_save("Importance Methods: SHAP and MDI", report)
        
        # Initialize Powerkit with specified split size
        pk = PowerkitWithVariableSplit(feature_data, label_data, cv_split_size=split_size)
        
        # Register conditions for all methods
        for condition in conditions:
            # Extract method name and importance method from condition string
            parts = condition.split('_')
            method_name = '_'.join(parts[2:-2])  # Extract method name part
            importance_method = parts[-1]  # Extract importance method
            
            pipeline_func = create_feature_importance_pipeline(
                methods[method_name], k_value, method_name, model_name, importance_method
            )
            
            # Add condition to Powerkit
            pk.add_condition(
                condition=condition,
                get_importance=True,
                pipeline_function=pipeline_func,
                pipeline_args={},
                eval_function=feature_importance_eval,
                eval_args={"metric_primary": "r2"}
            )
        
        print_and_save(f"Registered {len(pk.conditions)} conditions", report)
        
        print_and_save("## Consensus Run Execution", report)
        print_and_save("Starting consensus runs for feature selection method comparison...", report)
        
        # Run consensus analysis for each condition
        all_consensus_results = {}
        all_iteration_results = {}
        
        for condition in conditions:
            print_and_save(f"### Running consensus for condition: {condition}", report)
            
            consensus_start = time.time()
            
            # Run until consensus with specified parameters
            rng_list, total_df, meta_df = pk.run_until_consensus(
                condition=condition,
                rel_tol=0.01,
                abs_tol=0.001,
                max_iter=250,
                n_jobs=-1,
                verbose=True,
                verbose_level=1,
                return_meta_df=True,
                crunch_factor=2
            )
            
            consensus_time = time.time() - consensus_start
            
            print_and_save(f"Consensus run completed in {consensus_time:.2f} seconds", report)
            print_and_save(f"Number of iterations: {len(rng_list)}", report)
            print_and_save(f"Final relative tolerance: {meta_df['current_tol'].iloc[-1]:.6f}", report)
            print_and_save(f"Final absolute difference: {meta_df['abs_diff'].iloc[-1]:.6f}", report)
            
            # Save consensus feature importance
            consensus_importance, iteration_importance, consensus_signed_importance, iteration_signed_importance = save_consensus_feature_importance(
                total_df, condition, file_save_path
            )
            
            all_consensus_results[condition] = consensus_importance
            all_iteration_results[condition] = iteration_importance
            
            # Report on signed importance availability
            if consensus_signed_importance is not None:
                print_and_save(f"✓ Signed feature importance saved for {condition}", report)
            else:
                print_and_save(f"✓ Only absolute feature importance available for {condition} (MDI method)", report)
            
            # Save main results
            total_df.to_pickle(f"{file_save_path}total_results_{condition}.pkl")
            meta_df.to_pickle(f"{file_save_path}meta_results_{condition}.pkl")
            
            print_and_save(f"Results saved for condition: {condition}", report)
        
        print_and_save("## Feature Selection Method Comparison", report)
        
        # Compare feature selection methods (using SHAP importance for comparison)
        shap_results = {}
        for condition, consensus_df in all_consensus_results.items():
            if 'shap' in condition:
                method_name = '_'.join(condition.split('_')[2:-2])
                shap_results[method_name] = consensus_df
        
        if len(shap_results) == 4:  # All 4 methods should be present
            comparison_df = compare_feature_selection_methods(
                shap_results, file_save_path, f"k{k_value}_split{split_size}"
            )
            
            print_and_save("### Top 10 Features by Average Importance Across Methods", report)
            top_features = comparison_df.head(10)
            print_and_save(top_features[['feature_name', 'avg_importance'] + 
                                      [f'{method}_mean_importance' for method in shap_results.keys()]].to_string(), report)
            
            print_and_save("### Method Performance Summary", report)
            for method_name, consensus_df in shap_results.items():
                top_feature = consensus_df.iloc[0] if len(consensus_df) > 0 else None
                if top_feature is not None:
                    print_and_save(f"{method_name}: Top feature importance = {top_feature['mean_importance']:.6f} ± {top_feature['std_importance']:.6f}", report)
                else:
                    print_and_save(f"{method_name}: No features selected", report)
            
            print_and_save("### Feature Set Overlap Analysis", report)
            overlap_df = analyze_feature_set_overlap(
                shap_results, file_save_path, f"k{k_value}_split{split_size}"
            )
            
            print_and_save("#### Pairwise Feature Set Overlap", report)
            for _, row in overlap_df.iterrows():
                if row['method_pair'] != 'all_methods':
                    print_and_save(f"{row['method_pair']}:", report)
                    print_and_save(f"  - Method 1 features: {row['method1_features']}", report)
                    print_and_save(f"  - Method 2 features: {row['method2_features']}", report)
                    print_and_save(f"  - Overlap count: {row['overlap_count']}", report)
                    print_and_save(f"  - Jaccard similarity: {row['jaccard_similarity']:.4f}", report)
                    print_and_save(f"  - Overlap percentage (method 1): {row['overlap_percentage_method1']:.2%}", report)
                    print_and_save(f"  - Overlap percentage (method 2): {row['overlap_percentage_method2']:.2%}", report)
            
            print_and_save("#### Overall Feature Set Overlap", report)
            all_methods_row = overlap_df[overlap_df['method_pair'] == 'all_methods'].iloc[0]
            print_and_save(f"Total unique features across all methods: {all_methods_row['method1_features']}", report)
            print_and_save(f"Features common to all methods: {all_methods_row['overlap_count']}", report)
            print_and_save(f"Overall Jaccard similarity: {all_methods_row['jaccard_similarity']:.4f}", report)
            print_and_save(f"Percentage of total features common to all methods: {all_methods_row['overlap_percentage_method1']:.2%}", report)
        
        # Calculate total batch time
        total_batch_time = time.time() - total_batch_start
        
        print_and_save("## Total Execution Time", report)
        print_and_save(f"Total execution time: {total_batch_time:.2f} seconds", report)
        
        print_and_save("## Conclusion", report)
        print_and_save("Feature selection method comparison completed successfully!", report)
        print_and_save(f"Report saved to: {report_file}", report)
        print_and_save(f"Results saved to: {file_save_path}", report)
        print_and_save("Key files generated:", report)
        print_and_save("- Consensus feature importance files for each method (*_consensus_feature_importance_*.pkl)", report)
        print_and_save("- Iteration-level importance files (*_iteration_feature_importance_*.pkl)", report)
        print_and_save("- Method comparison file (*_feature_selection_method_comparison_*.pkl)", report)
        print_and_save("- Main results files (*_total_results_*.pkl)", report)
        print_and_save("- Meta results files (*_meta_results_*.pkl)", report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
