#!/usr/bin/env python3
"""
Batch script for feature selection benchmarking
Converts the Jupyter notebook to an executable batch script
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Literal
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

# Import custom modules
from PathLoader import PathLoader
from DataLink import DataLink
from toolkit import (
    FirstQuantileImputer, 
    f_regression_select, 
    get_model_from_string,
    greedy_feedforward_select,
    mrmr_select_fcq_fast,
    mutual_information_select,
    select_random_features,
    Powerkit
)


def random_select_wrapper(X: pd.DataFrame, y: pd.Series, k: int) -> tuple:
    """Wrapper function for random feature selection that returns dummy scores"""
    selected_features, _ = select_random_features(X, y, k)
    # Return dummy scores (all zeros) since random selection has no meaningful scores
    dummy_scores = np.zeros(len(selected_features))
    return selected_features, dummy_scores


def get_most_correlated_feature(X: pd.DataFrame, y: pd.Series) -> str:
    """Find the feature with highest correlation to target (from gffs_timing_test.py)"""
    correlations = []
    for col in X.columns:
        corr = np.corrcoef(X[col], y)[0, 1]
        correlations.append((col, abs(corr)))
    
    # Return feature with highest absolute correlation
    most_correlated = max(correlations, key=lambda x: x[1])
    return most_correlated[0]


def gffs_select_wrapper(X: pd.DataFrame, y: pd.Series, k: int) -> tuple:
    """Wrapper function for GFFS feature selection that handles setup automatically"""
    from sklearn.svm import SVR
    
    # Create SVR model with linear kernel (same as timing test)
    svr_model = SVR(kernel='linear', C=1.0)
    
    # Find starting feature (most correlated with target)
    start_feature = get_most_correlated_feature(X, y)
    
    # Run GFFS with same parameters as timing test
    selected_features, scores = greedy_feedforward_select(
        X, y, k, svr_model, start_feature, cv=5, scoring_method='r2', verbose=0
    )
    
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


def create_feature_selection_pipeline(
    selection_method: callable, k: int, method_name: str, model_name: str
):
    """Create pipeline for feature selection methods"""

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

        # 5) Standardization and model training
        if no_features or len(selected_features) == 0:
            model = DummyRegressor(strategy="mean")
            model_type = "DummyRegressor(mean)"
            model_params = {"strategy": "mean"}
            sel_train = Xtr_filtered.iloc[:, :0]
        else:
            sel_train = Xtr_filtered[selected_features]
            scaler = StandardScaler()
            sel_train_scaled = scaler.fit_transform(sel_train)
            sel_train_scaled = pd.DataFrame(
                sel_train_scaled, index=sel_train.index, columns=selected_features
            )

            # Train model
            if model_name == "LinearRegression":
                model = get_model_from_string("LinearRegression")
            elif model_name == "KNeighborsRegressor":
                model = get_model_from_string(
                    "KNeighborsRegressor", n_neighbors=5, weights="distance", p=2
                )
            elif model_name == "SVR":
                model = get_model_from_string("SVR", kernel="linear", C=1.0)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            model.fit(sel_train_scaled, y_train)
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
            "scaler": scaler if not no_features else None,
            "no_features": no_features,
            "rng": rng,
            "feature_selection_time": feature_selection_time,
        }

    return pipeline_function


def feature_selection_eval(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    pipeline_components: Dict,
    metric_primary: Literal["r2", "pearson_r", "spearman_r"] = "r2",
) -> Dict:
    """Evaluation function for feature selection benchmarking"""

    # Unpack components following the structure from working baseline code
    imputer = pipeline_components["imputer"]
    corr_keep = set(pipeline_components["corr_keep_cols"])
    selected = list(pipeline_components["selected_features"])
    selector_scores = pipeline_components["selector_scores"]
    model = pipeline_components["model"]
    model_name = pipeline_components["model_type"]
    scaler = pipeline_components.get("scaler", None)
    no_features = pipeline_components.get("no_features", False)
    feature_selection_time = pipeline_components.get("feature_selection_time", 0.0)

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

    # Predict
    if no_features or Xsel.shape[1] == 0:
        y_pred = np.full_like(
            y_test.values, fill_value=float(y_test.mean()), dtype=float
        )
    else:
        y_pred = np.asarray(model.predict(Xsel_scaled), dtype=float)

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

    # Feature importance
    if not no_features and hasattr(model, "feature_importances_") and len(selected) > 0:
        fi = (np.array(selected), model.feature_importances_)
    elif not no_features and model_name in ("LinearRegression",) and len(selected) > 0:
        coef = getattr(model, "coef_", np.zeros(len(selected)))
        fi = (np.array(selected), np.abs(coef))
    else:
        fi = (np.array(selected), np.zeros(len(selected)))

    primary = metrics.get(metric_primary, metrics["r2"])

    return {
        "feature_importance": fi,
        "feature_importance_from": "model",
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
    }


def main():
    """Main execution function for feature selection benchmarking"""
    # Initialize path loader and data link
    path_loader = PathLoader("data_config.env", "current_user.env")
    data_link = DataLink(path_loader, "data_codes.csv")
    
    # Setup experiment parameters
    folder_name = "ThesisResult4-FeatureSelectionBenchmark"
    exp_id = "v4_gffs_included"
    
    # Create results directory
    if not os.path.exists(f"{path_loader.get_data_path()}data/results/{folder_name}"):
        os.makedirs(f"{path_loader.get_data_path()}data/results/{folder_name}")
    
    file_save_path = f"{path_loader.get_data_path()}data/results/{folder_name}/"
    
    # Create report file
    report_file = f"{file_save_path}feature_selection_benchmark_report_{exp_id}.md"
    
    def print_and_save(message, file_handle):
        print(message)
        file_handle.write(message + "\n")
    
    with open(report_file, 'w', encoding='utf-8') as report:
        print_and_save("# Feature Selection Benchmarking Report", report)
        print_and_save(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}", report)
        print_and_save(f"**Experiment ID:** {exp_id}", report)
        print_and_save("", report)
        
        print_and_save("## Execution Summary", report)
        print_and_save("Starting feature selection benchmarking batch script...", report)
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
        
        print_and_save("## Experiment Setup", report)
        # Setup experiment parameters
        feature_set_sizes = [5, 10, 20, 40]  # Limited to focus on most relevant sizes
        models = ["KNeighborsRegressor", "LinearRegression", "SVR"]
        
        # Define the feature selection methods including random selection as negative control
        feature_selection_methods = {
            "anova_filter": f_regression_select,        # Renamed as ANOVA-filter
            "mrmr": mrmr_select_fcq_fast,              # MRMR method
            "mutual_info": mutual_information_select,  # Mutual Information method
            "gffs": gffs_select_wrapper,               # GFFS method (new addition)
            "random_select": random_select_wrapper     # Random selection as negative control
        }
        
        print_and_save(f"Benchmarking {len(feature_selection_methods)} methods across {len(feature_set_sizes)} feature sizes and {len(models)} models", report)
        print_and_save(f"Total conditions: {len(feature_selection_methods) * len(feature_set_sizes) * len(models)}", report)
        print_and_save("Methods: ANOVA-filter, MRMR, Mutual Information, GFFS, and Random Selection (negative control)", report)
        
        print_and_save("## Powerkit Setup", report)
        # Initialize Powerkit with proteomics data
        pk = Powerkit(feature_data, label_data)
        
        # Register all conditions (method × size × model combinations)
        rngs = np.random.RandomState(42).randint(0, 100000, size=1)  # Single run for batch execution
        
        start_time = time.time()
        
        for method_name, selection_method in feature_selection_methods.items():
            for k in feature_set_sizes:
                for model_name in models:
                    # Create condition name using the requested naming convention
                    condition = f"{method_name}_k{k}_{model_name}"
                    
                    # Create pipeline for this method and size
                    pipeline_func = create_feature_selection_pipeline(selection_method, k, method_name, model_name)
                    
                    # Add condition to Powerkit
                    pk.add_condition(
                        condition=condition,
                        get_importance=True,
                        pipeline_function=pipeline_func,
                        pipeline_args={},
                        eval_function=feature_selection_eval,
                        eval_args={"metric_primary": "r2"}
                    )
        
        print_and_save(f"Registered {len(pk.conditions)} conditions in {time.time() - start_time:.2f} seconds", report)
        
        print_and_save("## Benchmark Execution", report)
        # Run all conditions using Powerkit's parallel processing
        print_and_save("Starting feature selection benchmark...", report)
        print_and_save(f"Running with {len(rngs)} random seeds and -1 n_jobs for maximum parallelization", report)
        
        benchmark_start = time.time()
        df_benchmark = pk.run_all_conditions(rng_list=rngs, n_jobs=-1, verbose=True)
        benchmark_time = time.time() - benchmark_start
        
        print_and_save(f"Benchmark completed in {benchmark_time:.2f} seconds", report)
        print_and_save(f"Results shape: {df_benchmark.shape}", report)
        
        # Extract k value and model name from condition for easier analysis
        df_benchmark["k_value"] = df_benchmark["condition"].str.extract(r'k(\d+)').astype(int)
        df_benchmark["method"] = df_benchmark["condition"].str.split('_').str[0]
        df_benchmark["model_name"] = df_benchmark["condition"].str.split('_').str[2]
        
        print_and_save("## Results", report)
        print_and_save("### Condition Breakdown", report)
        print_and_save("Condition breakdown:", report)
        print_and_save(df_benchmark[["condition", "method", "k_value", "model_name"]].head(10).to_string(), report)
        
        # Save results
        df_benchmark.to_pickle(f"{file_save_path}feature_selection_benchmark_{exp_id}.pkl")
        print_and_save(f"Results saved to: {file_save_path}feature_selection_benchmark_{exp_id}.pkl", report)
        
        print_and_save("### Performance Summary", report)
        print_and_save("Benchmark Summary:", report)
        print_and_save(f"Total runs: {len(df_benchmark)}", report)
        print_and_save(f"Unique conditions: {df_benchmark['condition'].nunique()}", report)
        print_and_save(f"Performance range (R²): {df_benchmark['model_performance'].min():.4f} to {df_benchmark['model_performance'].max():.4f}", report)

        # Show performance by method (including random selection as negative control)
        method_summary = df_benchmark.groupby("method")["model_performance"].agg(["mean", "std", "count"])
        print_and_save("\nPerformance by method (ANOVA-filter, MRMR, Mutual Information, GFFS, Random Selection):", report)
        print_and_save(method_summary.round(4).to_string(), report)

        print_and_save("### Feature Selection Time Analysis", report)
        print_and_save("Feature Selection Time Analysis:", report)
        print_and_save(f"Feature selection time range: {df_benchmark['feature_selection_time'].min():.6f} to {df_benchmark['feature_selection_time'].max():.6f} seconds", report)
        
        # Time analysis by method
        time_summary = df_benchmark.groupby("method")["feature_selection_time"].agg(["mean", "std", "count"])
        print_and_save("\nFeature Selection Time by Method:", report)
        print_and_save(time_summary.round(6).to_string(), report)
        
        # Time analysis by k value
        k_time_summary = df_benchmark.groupby("k_value")["feature_selection_time"].agg(["mean", "std", "count"])
        print_and_save("\nFeature Selection Time by k Value:", report)
        print_and_save(k_time_summary.round(6).to_string(), report)
        
        # Time vs. performance correlation
        time_perf_corr = df_benchmark["feature_selection_time"].corr(df_benchmark["model_performance"])
        print_and_save(f"\nCorrelation between feature selection time and performance (R²): {time_perf_corr:.4f}", report)

        print_and_save("## Conclusion", report)
        print_and_save("Feature selection benchmarking completed successfully!", report)
        print_and_save(f"Report saved to: {report_file}", report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred during execution: {e}")
        sys.exit(1)
