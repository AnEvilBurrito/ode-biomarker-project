#!/usr/bin/env python3
"""
Feature Importance Analysis with Random Forest and SHAP for Network-Specific Gene Subsets
Focuses on comparing feature importance patterns across network-specific gene subsets vs dynamic features
Uses only RandomForestRegressor config1 model with SHAP importance evaluation
"""

import os
import sys
import time
import gc  # Memory cleanup
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


# Network-specific gene lists (same as in benchmark script)
fgfr4_genes = [
    "IGF1R",
    "FGFR4",
    "ERBB3", "ERBB2",
    "IRS1", "IRS2",
    "PIK3CA", "PIK3CD", "PIK3CB", "PIK3CG",
    "FRS2",
    "GRB2",
    "AKT1", "AKT2", "AKT3",
    "PDK1",
    "RPTOR",
    "RPS6KB1",
    "SOS2", "SOS1",
    "PTPN11",
    "NRAS", "KRAS", "HRAS",
    "RAF1", "ARAF", "BRAF",
    "MAP2K1", "MAP2K2",
    "MAPK1", "MAPK3",
    "GAB1",
    "GAB2",
    "SPRY2",
    "PTPN12",
    "CBL",
    "FOXO3",
    "RICTOR"
]

cdk46_genes = [
    "AKT3", "AKT2", "AKT1",
    "CDK2",
    "CDK6", "CDK4",
    "CCND1", "CCND2", "CCND3",
    "CCNE1", "CCNE2",
    "E2F8", "E2F7", "E2F6",
    "MAPK7", "MAPK8", "MAPK9",
    "GSK3B",
    "INSR",
    "IRS4", "IRS2", "IRS1",
    "RPTOR",
    "RICTOR",
    "MYC",
    "CDKN1A",
    "CDKN1B",
    "PDK1",
    "PIK3CA", "PIK3CD", "PIK3CB", "PIK3CG",
    "RAF1", "BRAF", "ARAF",
    "RB1",
    "RPS6KB1",
    "SOS2", "SOS1"
]


def extract_gene_subset(feature_data: pd.DataFrame, gene_list: List[str]) -> pd.DataFrame:
    """
    Extract specific gene columns from RNASeq feature data
    
    Parameters:
    -----------
    feature_data: Full RNASeq feature dataframe
    gene_list: List of gene names to extract
    
    Returns:
    --------
    Filtered dataframe containing only the specified genes
    """
    # Find matching columns (case-insensitive matching)
    available_genes = set(feature_data.columns)
    filtered_genes = []
    
    for gene in gene_list:
        # Try exact match first
        if gene in available_genes:
            filtered_genes.append(gene)
        else:
            # Try case-insensitive matching
            matches = [col for col in available_genes if col.upper() == gene.upper()]
            if matches:
                filtered_genes.extend(matches)
    
    print(f"Extracted {len(filtered_genes)} genes out of {len(gene_list)} requested")
    print(f"Missing genes: {set(gene_list) - set(filtered_genes)}")
    
    return feature_data[filtered_genes]


def mrmr_standard_select(X: pd.DataFrame, y: pd.Series, k: int) -> tuple:
    """
    Standard MRMR feature selection wrapper
    Simplified version without network constraints
    """
    # Cap k at available features
    k_actual = min(k, X.shape[1])
    
    if k_actual == 0:
        return [], np.array([])
    
    # Apply standard MRMR
    selected_features, scores = mrmr_select_fcq_fast(X, y, k_actual, verbose=0)
    
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


def create_shap_feature_importance_pipeline(
    selection_method: callable, k: int, method_name: str, model_name: str
):
    """
    Create pipeline for Random Forest model with SHAP feature importance
    Focuses on RandomForestRegressor config1 only
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
        scaler = None
        
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

            # Train model with timing - Only RandomForest config1
            model_training_start = time.time()
            model = get_model_from_string("RandomForestRegressor", n_estimators=100, random_state=rng)
            
            model.fit(sel_train_scaled, y_train)
            model_training_time = time.time() - model_training_start
            model_type = model_name
            model_params = model.get_params(deep=False) if hasattr(model, "get_params") else {}

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
            "importance_method": "shap",  # Force SHAP importance
        }

    return pipeline_function


def shap_feature_importance_eval(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    pipeline_components: Dict,
    metric_primary: Literal["r2", "pearson_r", "spearman_r"] = "r2",
) -> Dict:
    """
    Evaluation function for SHAP feature importance analysis
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

    # Standardize if scaler exists
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

    # SHAP Feature importance calculation
    fi_absolute = None
    fi_signed = None
    
    if not no_features and len(selected) > 0:
        # Calculate SHAP values
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xsel_scaled)
            
            # Handle different SHAP value formats
            if len(shap_values.shape) == 2:
                # For regression, shap_values might be 2D
                feature_importance_abs = np.abs(shap_values).mean(axis=0)
                feature_importance_signed = shap_values.mean(axis=0)
            else:
                # For classification or single output
                feature_importance_abs = np.abs(shap_values)
                feature_importance_signed = shap_values
            
            fi_absolute = (np.array(selected), feature_importance_abs)
            fi_signed = (np.array(selected), feature_importance_signed)
            importance_source = "shap"
        except Exception as e:
            print(f"SHAP calculation failed: {e}")
            # Fall back to MDI if SHAP fails
            if hasattr(model, "feature_importances_"):
                fi_absolute = (np.array(selected), model.feature_importances_)
                fi_signed = None
            else:
                fi_absolute = (np.array(selected), np.zeros(len(selected)))
                fi_signed = None
            importance_source = "mdi_fallback"
    else:
        fi_absolute = (np.array(selected), np.zeros(len(selected)))
        fi_signed = None
        importance_source = "none"

    primary = metrics.get(metric_primary, metrics["r2"])

    return {
        "feature_importance": fi_absolute,  # Absolute SHAP values (backward compatible)
        "feature_importance_signed": fi_signed,  # Signed SHAP values
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
        "importance_method": "shap",
    }


def save_shap_feature_importance_analysis(total_df, file_save_path, condition_base):
    """
    Save comprehensive SHAP feature importance analysis results
    Includes both absolute and signed importance values
    """
    # Extract SHAP importance data for all conditions
    all_importance_data = []
    all_signed_importance_data = []
    
    for condition in total_df['condition'].unique():
        condition_df = total_df[total_df['condition'] == condition]
        
        # Extract absolute importance
        for _, row in condition_df.iterrows():
            feature_names = row['feature_importance'][0]
            importance_scores = row['feature_importance'][1]
            rng = row['rng']
            
            for feature_name, score in zip(feature_names, importance_scores):
                all_importance_data.append({
                    'condition': condition,
                    'iteration_rng': rng,
                    'feature_name': feature_name,
                    'importance_score': score,
                    'importance_method': 'shap'
                })
        
        # Extract signed importance if available
        for _, row in condition_df.iterrows():
            if row.get('feature_importance_signed') is not None:
                feature_names = row['feature_importance_signed'][0]
                signed_scores = row['feature_importance_signed'][1]
                rng = row['rng']
                
                for feature_name, score in zip(feature_names, signed_scores):
                    all_signed_importance_data.append({
                        'condition': condition,
                        'iteration_rng': rng,
                        'feature_name': feature_name,
                        'importance_score_signed': score,
                        'importance_method': 'shap'
                    })
    
    # Create DataFrames
    if all_importance_data:
        iteration_importance_df = pd.DataFrame(all_importance_data)
        consensus_importance = iteration_importance_df.groupby(['condition', 'feature_name']).agg({
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
        
        # Save absolute importance results
        consensus_importance.to_pickle(f"{file_save_path}shap_consensus_importance_{condition_base}.pkl")
        iteration_importance_df.to_pickle(f"{file_save_path}shap_iteration_importance_{condition_base}.pkl")
    
    if all_signed_importance_data:
        iteration_signed_importance_df = pd.DataFrame(all_signed_importance_data)
        consensus_signed_importance = iteration_signed_importance_df.groupby(['condition', 'feature_name']).agg({
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
        
        # Save signed importance results
        consensus_signed_importance.to_pickle(f"{file_save_path}shap_consensus_importance_signed_{condition_base}.pkl")
        iteration_signed_importance_df.to_pickle(f"{file_save_path}shap_iteration_importance_signed_{condition_base}.pkl")
    
    return iteration_importance_df if all_importance_data else None


def analyze_dataset_type_importance_patterns(total_df, file_save_path, condition_base):
    """
    Analyze feature importance patterns across different dataset types
    """
    # Extract dataset type and network information from condition names
    dataset_analysis = []
    
    for condition in total_df['condition'].unique():
        # Parse condition name to extract dataset type
        parts = condition.split('_')
        network_type = parts[-2] if len(parts) > 4 else 'unknown'
        dataset_type = parts[-1] if len(parts) > 4 else 'unknown'
        
        condition_df = total_df[total_df['condition'] == condition]
        performance_mean = condition_df['model_performance'].mean()
        
        # Extract top features
        if len(condition_df) > 0:
            # Get average importance for top 10 features
            feature_importance_data = []
            for _, row in condition_df.iterrows():
                feature_names = row['feature_importance'][0]
                importance_scores = row['feature_importance'][1]
                
                for feature_name, score in zip(feature_names, importance_scores):
                    feature_importance_data.append({
                        'feature_name': feature_name,
                        'importance_score': score
                    })
            
            if feature_importance_data:
                importance_df = pd.DataFrame(feature_importance_data)
                top_features = importance_df.groupby('feature_name')['importance_score'].mean().nlargest(10)
                
                dataset_analysis.append({
                    'condition': condition,
                    'network_type': network_type,
                    'dataset_type': dataset_type,
                    'performance_mean': performance_mean,
                    'top_features': list(top_features.index),
                    'top_importance_scores': list(top_features.values)
                })
    
    dataset_analysis_df = pd.DataFrame(dataset_analysis)
    dataset_analysis_df.to_pickle(f"{file_save_path}dataset_type_analysis_{condition_base}.pkl")
    
    return dataset_analysis_df


def main():
    """Main execution function for feature importance analysis with gene subsets"""
    # Configuration parameters
    TOTAL_SEEDS = 20
    SEEDS_PER_BATCH = 4
    
    # Fixed RNG seeds (same as benchmark script)
    SEEDS = np.array([15795, 860, 76820, 54886, 6265, 82386, 37194, 87498, 44131, 60263, 16023, 41090, 67221, 64820, 769, 59735, 62955, 64925, 67969, 5311])
    
    # Initialize path loader and data link
    path_loader = PathLoader("data_config.env", "current_user.env")
    data_link = DataLink(path_loader, "data_codes.csv")
    
    # Setup experiment parameters
    folder_name = "ThesisResult-FeatureImportanceGeneSubsets-SHAP"
    exp_id = f"v1_rf_config1_genesubsets_shap_seeds{TOTAL_SEEDS}_batch{SEEDS_PER_BATCH}"
    
    # Create results directory
    if not os.path.exists(f"{path_loader.get_data_path()}data/results/{folder_name}"):
        os.makedirs(f"{path_loader.get_data_path()}data/results/{folder_name}")
    
    file_save_path = f"{path_loader.get_data_path()}data/results/{folder_name}/"
    
    # Create report file
    report_file = f"{file_save_path}feature_importance_analysis_report_{exp_id}.md"
    
    def print_and_save(message, file_handle):
        print(message)
        file_handle.write(message + "\n")
    
    # Start total batch timing
    total_batch_start = time.time()
    
    with open(report_file, 'w', encoding='utf-8') as report:
        print_and_save("# Feature Importance Analysis Report (Random Forest + SHAP - Gene Subsets)", report)
        print_and_save(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}", report)
        print_and_save(f"**Experiment ID:** {exp_id}", report)
        print_and_save(f"**Total Seeds:** {TOTAL_SEEDS}", report)
        print_and_save(f"**Seeds per Batch:** {SEEDS_PER_BATCH}", report)
        print_and_save(f"**Fixed RNG Seeds:** Using provided seed array", report)
        print_and_save("", report)
        
        print_and_save("## Execution Summary", report)
        print_and_save("Analyzing feature importance patterns across network-specific gene subsets using Random Forest config1 and SHAP...", report)
        print_and_save(f"Results will be saved to: {file_save_path}", report)
        print_and_save("**Key difference:** Using network-specific gene subsets instead of full RNASeq", report)
    
    with open(report_file, 'a', encoding='utf-8') as report:
        print_and_save("## Data Loading and Preparation", report)
        
        # Load CDK4/6 datasets
        print_and_save("### Loading CDK4/6 Network Datasets", report)
        
        # Load CDK4/6 full RNASeq dataset and extract specific genes
        loading_code = "ccle-gdsc-2-Palbociclib-LN_IC50"
        cdk46_full_rnaseq_feature_data, cdk46_rnaseq_label_data = data_link.get_data_using_code(loading_code)
        print_and_save(f"CDK4/6 Full RNASeq feature data shape: {cdk46_full_rnaseq_feature_data.shape}", report)
        
        # Extract network-specific genes for CDK4/6
        cdk46_rnaseq_feature_data = extract_gene_subset(cdk46_full_rnaseq_feature_data, cdk46_genes)
        print_and_save(f"CDK4/6 Gene Subset feature data shape: {cdk46_rnaseq_feature_data.shape}", report)
        
        # Load CDK4/6 dynamic dataset
        cdk46_feature_data_dynamic, cdk46_label_data_dynamic = data_link.get_data_using_code('generic-gdsc-2-Palbociclib-LN_IC50-cdk46_ccle_dynamic_features_v4_ccle-true-Unnamed: 0')
        print_and_save(f"CDK4/6 Dynamic feature data shape: {cdk46_feature_data_dynamic.shape}", report)
        
        # Load FGFR4 datasets
        print_and_save("### Loading FGFR4 Network Datasets", report)
        
        # Load FGFR4 full RNASeq dataset and extract specific genes
        loading_code = "ccle-gdsc-1-FGFR_0939-LN_IC50"
        fgfr4_full_rnaseq_feature_data, fgfr4_rnaseq_label_data = data_link.get_data_using_code(loading_code)
        print_and_save(f"FGFR4 Full RNASeq feature data shape: {fgfr4_full_rnaseq_feature_data.shape}", report)
        
        # Extract network-specific genes for FGFR4
        fgfr4_rnaseq_feature_data = extract_gene_subset(fgfr4_full_rnaseq_feature_data, fgfr4_genes)
        print_and_save(f"FGFR4 Gene Subset feature data shape: {fgfr4_rnaseq_feature_data.shape}", report)
        
        # Load FGFR4 dynamic dataset
        fgfr4_feature_data_dynamic, fgfr4_label_data_dynamic = data_link.get_data_using_code('generic-gdsc-1-FGFR_0939-LN_IC50-fgfr4_ccle_dynamic_features_v2-true-Unnamed: 0')
        print_and_save(f"FGFR4 Dynamic feature data shape: {fgfr4_feature_data_dynamic.shape}", report)
        
        print_and_save("## Data Preparation and Alignment", report)
        
        # Prepare datasets dictionary for systematic processing
        datasets = {}
        
        # CDK4/6 Network datasets
        datasets['cdk46_genesubset'] = {
            'features': cdk46_rnaseq_feature_data.select_dtypes(include=[np.number]),
            'labels': cdk46_rnaseq_label_data,
            'network': 'cdk46',
            'type': 'genesubset'
        }
        
        datasets['cdk46_dynamic'] = {
            'features': cdk46_feature_data_dynamic.select_dtypes(include=[np.number]),
            'labels': cdk46_label_data_dynamic,
            'network': 'cdk46',
            'type': 'dynamic'
        }
        
        # FGFR4 Network datasets
        datasets['fgfr4_genesubset'] = {
            'features': fgfr4_rnaseq_feature_data.select_dtypes(include=[np.number]),
            'labels': fgfr4_rnaseq_label_data,
            'network': 'fgfr4',
            'type': 'genesubset'
        }
        
        datasets['fgfr4_dynamic'] = {
            'features': fgfr4_feature_data_dynamic.select_dtypes(include=[np.number]),
            'labels': fgfr4_label_data_dynamic,
            'network': 'fgfr4',
            'type': 'dynamic'
        }
        
        # Create combined datasets
        print_and_save("### Creating Combined Datasets", report)
        
        # CDK4/6 combined
        common_cdk46_indices = sorted(
            set(cdk46_rnaseq_feature_data.index) & set(cdk46_feature_data_dynamic.index)
        )
        if len(common_cdk46_indices) > 0:
            cdk46_rnaseq_aligned = cdk46_rnaseq_feature_data.loc[common_cdk46_indices].select_dtypes(include=[np.number])
            cdk46_dynamic_aligned = cdk46_feature_data_dynamic.loc[common_cdk46_indices].select_dtypes(include=[np.number])
            
            # Combine features, ensuring unique column names
            cdk46_combined_features = pd.concat([cdk46_rnaseq_aligned, cdk46_dynamic_aligned], axis=1)
            cdk46_combined_labels = cdk46_rnaseq_label_data.loc[common_cdk46_indices]
            
            datasets['cdk46_combined'] = {
                'features': cdk46_combined_features,
                'labels': cdk46_combined_labels,
                'network': 'cdk46',
                'type': 'combined'
            }
            print_and_save(f"CDK4/6 Combined dataset shape: {cdk46_combined_features.shape}", report)
        else:
            print_and_save("Warning: No common samples for CDK4/6 combined dataset", report)
        
        # FGFR4 combined
        common_fgfr4_indices = sorted(
            set(fgfr4_rnaseq_feature_data.index) & set(fgfr4_feature_data_dynamic.index)
        )
        if len(common_fgfr4_indices) > 0:
            fgfr4_rnaseq_aligned = fgfr4_rnaseq_feature_data.loc[common_fgfr4_indices].select_dtypes(include=[np.number])
            fgfr4_dynamic_aligned = fgfr4_feature_data_dynamic.loc[common_fgfr4_indices].select_dtypes(include=[np.number])
            
            # Combine features, ensuring unique column names
            fgfr4_combined_features = pd.concat([fgfr4_rnaseq_aligned, fgfr4_dynamic_aligned], axis=1)
            fgfr4_combined_labels = fgfr4_rnaseq_label_data.loc[common_fgfr4_indices]
            
            datasets['fgfr4_combined'] = {
                'features': fgfr4_combined_features,
                'labels': fgfr4_combined_labels,
                'network': 'fgfr4',
                'type': 'combined'
            }
            print_and_save(f"FGFR4 Combined dataset shape: {fgfr4_combined_features.shape}", report)
        else:
            print_and_save("Warning: No common samples for FGFR4 combined dataset", report)
        
        print_and_save("## Experiment Setup", report)
        # Setup experiment parameters - Focus on RandomForest config1 only
        feature_set_sizes = [500]
        model_name = "RandomForestRegressor_config1"  # Only this model
        
        # Define the single feature selection method (standard MRMR)
        feature_selection_methods = {}
        method_name = "mrmr"
        feature_selection_methods[method_name] = mrmr_standard_select
        
        print_and_save(f"Benchmarking {len(datasets)} datasets across {len(feature_set_sizes)} feature sizes", report)
        print_and_save(f"Total conditions: {len(datasets) * len(feature_set_sizes)}", report)
        print_and_save(f"Total seeds: {TOTAL_SEEDS}", report)
        print_and_save(f"Seeds per batch: {SEEDS_PER_BATCH}", report)
        print_and_save(f"Gene subsets: FGFR4={len(fgfr4_genes)} genes, CDK4/6={len(cdk46_genes)} genes", report)
        print_and_save("Model: RandomForestRegressor config1 (100 trees)", report)
        print_and_save("Feature Selection Method: Standard MRMR", report)
        print_and_save("Importance Method: SHAP", report)
        print_and_save("Feature Set Size: 500", report)
        
        # Use provided seeds
        rngs = list(SEEDS[:TOTAL_SEEDS])  # Use only the first TOTAL_SEEDS seeds
        
        # Initialize results accumulator
        all_results = []
        
        # Process each dataset separately
        for dataset_name, dataset_info in datasets.items():
            print_and_save(f"### Processing dataset: {dataset_name}", report)
            
            # Prepare data for this dataset
            feature_data = dataset_info['features']
            label_data = dataset_info['labels']
            network_type = dataset_info['network']
            dataset_type = dataset_info['type']
            
            # Align indices
            common_indices = sorted(
                set(feature_data.index) & set(label_data.index)
            )
            if len(common_indices) == 0:
                print_and_save(f"Warning: No common samples for dataset {dataset_name}, skipping", report)
                continue
                
            feature_data_aligned = feature_data.loc[common_indices]
            label_data_aligned = label_data.loc[common_indices]
            
            print_and_save(f"Aligned dataset shape: {feature_data_aligned.shape}", report)
            print_and_save(f"Aligned label shape: {label_data_aligned.shape}", report)
            
            # Initialize Powerkit for this dataset
            pk = Powerkit(feature_data_aligned, label_data_aligned)
            
            # Register conditions (only RandomForest config1)
            for method_name, selection_method in feature_selection_methods.items():
                for k in feature_set_sizes:
                    # Create condition name with network and dataset info
                    condition = f"{model_name}_k{k}_{method_name}_{network_type}_{dataset_type}"
                    
                    # Create pipeline for this method and size
                    pipeline_func = create_shap_feature_importance_pipeline(selection_method, k, method_name, model_name)
                    
                    # Add condition to Powerkit
                    pk.add_condition(
                        condition=condition,
                        get_importance=True,
                        pipeline_function=pipeline_func,
                        pipeline_args={},
                        eval_function=shap_feature_importance_eval,
                        eval_args={"metric_primary": "r2"}
                    )
            
            print_and_save(f"Registered {len(pk.conditions)} conditions for {dataset_name}", report)
            
            # Batched seed execution
            print_and_save(f"Starting batched benchmark for {dataset_name} with {len(rngs)} seeds in batches of {SEEDS_PER_BATCH}...", report)
            
            dataset_results = []
            
            # Process seeds in batches
            for batch_start in range(0, len(rngs), SEEDS_PER_BATCH):
                batch_end = min(batch_start + SEEDS_PER_BATCH, len(rngs))
                batch_rngs = rngs[batch_start:batch_end]
                batch_number = (batch_start // SEEDS_PER_BATCH) + 1
                total_batches = (len(rngs) + SEEDS_PER_BATCH - 1) // SEEDS_PER_BATCH
                
                print_and_save(f"  Batch {batch_number}/{total_batches} processing seeds {batch_start+1}-{batch_end}...", report)
                
                batch_start_time = time.time()
                
                # Run conditions for this batch of seeds
                df_batch = pk.run_all_conditions(rng_list=batch_rngs, n_jobs=-1, verbose=False)
                
                batch_time = time.time() - batch_start_time
                
                # Add network and dataset type information
                df_batch["network_type"] = network_type
                df_batch["dataset_type"] = dataset_type
                
                # Extract k value and model name from condition for easier analysis
                df_batch["k_value"] = df_batch["condition"].str.extract(r'k(\d+)').astype(int)
                df_batch["model_name"] = model_name  # All are the same
                df_batch["method"] = method_name
                
                # Append to dataset results
                dataset_results.append(df_batch)
                
                print_and_save(f"    Completed batch in {batch_time:.2f} seconds ({len(batch_rngs)} seeds)", report)
                
                # Memory cleanup after each batch
                del df_batch
                gc.collect()
            
            # Combine all seed results for this dataset
            if dataset_results:
                df_dataset = pd.concat(dataset_results, ignore_index=True)
                all_results.append(df_dataset)
                
                # Clean up dataset results
                del dataset_results
                gc.collect()
            
            print_and_save(f"Completed dataset {dataset_name} in {time.time() - total_batch_start:.2f} seconds total", report)
        
        # Combine all results
        if all_results:
            df_final = pd.concat(all_results, ignore_index=True)
            
            print_and_save("## Final Results Summary", report)
            print_and_save("### Overall Statistics", report)
            print_and_save(f"Total runs across all datasets: {len(df_final)}", report)
            print_and_save(f"Unique conditions: {df_final['condition'].nunique()}", report)
            print_and_save(f"Performance range (R²): {df_final['model_performance'].min():.4f} to {df_final['model_performance'].max():.4f}", report)
            
            # Save main results
            df_final.to_pickle(f"{file_save_path}feature_importance_analysis_{exp_id}.pkl")
            print_and_save(f"Main results saved to: {file_save_path}feature_importance_analysis_{exp_id}.pkl", report)
            
            print_and_save("## Feature Importance Analysis", report)
            print_and_save("### Saving SHAP Feature Importance Results", report)
            
            # Save comprehensive SHAP analysis
            save_shap_feature_importance_analysis(df_final, file_save_path, exp_id)
            print_and_save("SHAP feature importance results saved", report)
            
            print_and_save("### Dataset Type Analysis", report)
            # Analyze importance patterns across dataset types
            dataset_analysis_df = analyze_dataset_type_importance_patterns(df_final, file_save_path, exp_id)
            print_and_save("Dataset type analysis completed", report)
            
            print_and_save("### Performance by Dataset Type", report)
            # Performance by dataset type
            dataset_summary = df_final.groupby(["network_type", "dataset_type"])["model_performance"].agg(["mean", "std", "count"])
            print_and_save(dataset_summary.round(4).to_string(), report)
            
            print_and_save("### SHAP Importance Source Summary", report)
            # SHAP vs fallback summary
            importance_source_summary = df_final["feature_importance_from"].value_counts()
            print_and_save(importance_source_summary.to_string(), report)
            
        else:
            print_and_save("No results generated - all datasets had issues", report)
        
        # Calculate total batch time
        total_batch_time = time.time() - total_batch_start
        
        print_and_save("## Total Execution Time", report)
        print_and_save(f"Total execution time: {total_batch_time:.2f} seconds", report)
        
        print_and_save("## Conclusion", report)
        print_and_save("Feature importance analysis with Random Forest and SHAP for network-specific gene subsets completed successfully!", report)
        print_and_save(f"Report saved to: {report_file}", report)
        print_and_save("Key files generated:", report)
        print_and_save(f"- Main results: {file_save_path}feature_importance_analysis_{exp_id}.pkl", report)
        print_and_save(f"- SHAP consensus importance: {file_save_path}shap_consensus_importance_{exp_id}.pkl", report)
        print_and_save(f"- SHAP iteration importance: {file_save_path}shap_iteration_importance_{exp_id}.pkl", report)
        print_and_save(f"- Dataset type analysis: {file_save_path}dataset_type_analysis_{exp_id}.pkl", report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
