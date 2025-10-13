# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: ode-biomarker-project
#     language: python
#     name: python3
# ---

# %%
# Jupyter notebook in Jupytext format

# %% [markdown]
# ## Initialisation

# %%
import os

path = os.getcwd()
# find the string 'project' in the path, return index
index_project = path.find("project")
# slice the path from the index of 'project' to the end
project_path = path[: index_project + 7]
# set the working directory
os.chdir(project_path)
print(f"Project path set to: {os.getcwd()}")

# %%
from PathLoader import PathLoader #noqa: E402

path_loader = PathLoader("data_config.env", "current_user.env")

# %%
from DataLink import DataLink #noqa: E402

data_link = DataLink(path_loader, "data_codes.csv")

# %%
folder_name = "ThesisResult4-FeatureSelectionBenchmark"
exp_id = "v1"

if not os.path.exists(f"{path_loader.get_data_path()}data/results/{folder_name}"):
    os.makedirs(f"{path_loader.get_data_path()}data/results/{folder_name}")

file_save_path = f"{path_loader.get_data_path()}data/results/{folder_name}/"

# %%
# Load Proteomics Palbociclib dataset
loading_code = "goncalves-gdsc-2-Palbociclib-LN_IC50-sin"
proteomic_feature_data, proteomic_label_data = data_link.get_data_using_code(
    loading_code
)

print(f"Proteomic feature data shape: {proteomic_feature_data.shape}")
print(f"Proteomic label data shape: {proteomic_label_data.shape}")

# %%
# Data preparation and alignment
import numpy as np #noqa: E402

# Ensure numeric only
proteomic_feature_data = proteomic_feature_data.select_dtypes(include=[np.number])

# Align indices
common_indices = sorted(
    set(proteomic_feature_data.index) & set(proteomic_label_data.index)
)
feature_data = proteomic_feature_data.loc[common_indices]
label_data = proteomic_label_data.loc[common_indices]

print(f"Final aligned dataset shape: {feature_data.shape}")
print(f"Final aligned label shape: {label_data.shape}")

# %% [markdown]
# ## Functions

# %%
from typing import Dict, List, Literal #noqa: E402
import numpy as np #noqa: E402
import pandas as pd #noqa: E402
from scipy.stats import pearsonr, spearmanr #noqa: E402
from sklearn.metrics import r2_score #noqa: E402
from sklearn.dummy import DummyRegressor #noqa: E402
from sklearn.preprocessing import StandardScaler #noqa: E402
from toolkit import FirstQuantileImputer, f_regression_select, get_model_from_string #noqa: E402
from toolkit import (
    mrmr_select_fcq, 
    mutual_information_select, 
) #noqa: E402
import time #noqa: E402


# %%
def _drop_correlated_columns(X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """Drop highly correlated columns to reduce redundancy"""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in sorted(upper.columns):
        if col in to_drop:
            continue
        high_corr = upper.index[upper[col] > threshold].tolist()
        to_drop.update(high_corr)
    return [c for c in X.columns if c not in to_drop]


# %%
def create_feature_selection_pipeline(
    selection_method: callable, k: int, method_name: str, model_name: str
):
    """Create pipeline for feature selection methods"""

    def pipeline_function(X_train: pd.DataFrame, y_train: pd.Series, rng: int):
        # 1) Imputation
        imputer = FirstQuantileImputer().fit(X_train)
        X_train_imp = imputer.transform(X_train, return_df=True)

        # 2) Optional: variance threshold and correlation filtering
        vt_keep_cols = list(X_train_imp.columns)  # Keep all initially
        corr_keep_cols = _drop_correlated_columns(X_train_imp, threshold=0.95)
        X_train_filtered = X_train_imp[corr_keep_cols]

        # 3) Feature selection with timing
        start_time = time.time()
        selected_features, selector_scores = selection_method(
            X_train_filtered, y_train, k
        )
        selection_time = time.time() - start_time

        # 4) Standardization (mandatory for our model suite)
        scaler = StandardScaler()
        sel_train_scaled = scaler.fit_transform(X_train_filtered[selected_features])
        sel_train_scaled = pd.DataFrame(
            sel_train_scaled, index=X_train_filtered.index, columns=selected_features
        )

        # 5) Train model
        if len(selected_features) == 0:
            model = DummyRegressor(strategy="mean")
            model_type = "DummyRegressor(mean)"
            model_params = {"strategy": "mean"}
        else:
            # Get model based on model_name parameter
            if model_name == "LinearRegression":
                model = get_model_from_string("LinearRegression")
                model_params = {"fit_intercept": True}
            elif model_name == "KNeighborsRegressor":
                model = get_model_from_string(
                    "KNeighborsRegressor", n_neighbors=5, weights="distance", p=2
                )
                model_params = {"n_neighbors": 5, "weights": "distance", "p": 2}
            elif model_name == "SVR":
                model = get_model_from_string("SVR", kernel="linear", C=1.0)
                model_params = {"kernel": "linear", "C": 1.0}
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            model.fit(sel_train_scaled, y_train)
            model_type = model_name

        return {
            "imputer": imputer,
            "scaler": scaler,
            "selected_features": list(selected_features),
            "selector_scores": np.array(selector_scores),
            "selection_time": selection_time,
            "method_name": method_name,
            "model": model,
            "model_type": model_type,
            "model_params": model_params,
            "rng": rng,
        }

    return pipeline_function


# %%
# %%
def feature_selection_eval(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    pipeline_components: Dict,
    metric_primary: Literal["r2", "pearson_r", "spearman_r"] = "r2",
) -> Dict:
    """Evaluation function for feature selection benchmarking"""

    # Unpack pipeline components
    imputer = pipeline_components["imputer"]
    scaler = pipeline_components["scaler"]
    selected = pipeline_components["selected_features"]
    selection_time = pipeline_components["selection_time"]
    model = pipeline_components["model"]
    model_name = pipeline_components["model_type"]

    # Transform test data
    X_test_imp = imputer.transform(X_test, return_df=True)

    # Apply same filtering as in training BUT ensure features exist in test data
    corr_keep_cols = _drop_correlated_columns(X_test_imp, threshold=0.95)
    X_test_filtered = X_test_imp[corr_keep_cols]

    # KEY FIX: Only select features that actually exist in the filtered test data
    available_features = [f for f in selected if f in X_test_filtered.columns]

    if len(available_features) == 0:
        # Fallback: if no selected features are available, use all filtered features
        available_features = X_test_filtered.columns.tolist()[
            : min(len(selected), X_test_filtered.shape[1])
        ]

    # Select features and standardize
    X_test_sel = (
        X_test_filtered[available_features]
        if len(available_features) > 0
        else X_test_filtered.iloc[:, :0]
    )

    if len(available_features) > 0:
        X_test_scaled = scaler.transform(X_test_sel)
        X_test_scaled = pd.DataFrame(
            X_test_scaled, index=X_test_filtered.index, columns=available_features
        )
    else:
        X_test_scaled = X_test_sel

    # Predict
    if len(available_features) == 0:
        y_pred = np.full_like(
            y_test.values, fill_value=float(y_test.mean()), dtype=float
        )
    else:
        y_pred = np.asarray(model.predict(X_test_scaled), dtype=float)

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

    # Feature importance (use available features only)
    if hasattr(model, "feature_importances_") and len(available_features) > 0:
        fi = (np.array(available_features), model.feature_importances_)
    elif model_name in ("LinearRegression",) and len(available_features) > 0:
        coef = getattr(model, "coef_", np.zeros(len(available_features)))
        fi = (np.array(available_features), np.abs(coef))
    else:
        fi = (np.array(available_features), np.zeros(len(available_features)))

    primary = metrics.get(metric_primary, metrics["r2"])

    return {
        "feature_importance": fi,
        "feature_importance_from": "model",
        "model_performance": float(primary) if primary is not None else np.nan,
        "metrics": metrics,
        "selection_time": selection_time,
        "n_features_selected": len(available_features),  # Use actual available count
        "model_name": model_name,
        "rng": pipeline_components.get("rng", None),
        "selected_features": available_features,  # Return what was actually used
        "selector_scores": pipeline_components.get("selector_scores", []),
        "y_pred": y_p,
        "y_true_index": y_test.index[mask_fin],
    }


# %% [markdown]
# ## Execution 

# %%
# Setup experiment parameters - using only the three requested methods
feature_set_sizes = [10, 20, 40, 80, 160, 320, 640, 1280]
models = ["KNeighborsRegressor", "LinearRegression", "SVR"]

# Define only the three feature selection methods you requested
feature_selection_methods = {
    "anova_filter": f_regression_select,        # Renamed as ANOVA-filter
    "mrmr": mrmr_select_fcq,                    # MRMR method [2]
    "mutual_info": mutual_information_select    # Mutual Information method [2]
}

print(f"Benchmarking {len(feature_selection_methods)} methods across {len(feature_set_sizes)} feature sizes and {len(models)} models")
print(f"Total conditions: {len(feature_selection_methods) * len(feature_set_sizes) * len(models)}")
print("Methods: ANOVA-filter, MRMR, and Mutual Information")

# %%
from toolkit import Powerkit
import numpy as np
import time #noqa: F811

# Initialize Powerkit with proteomics data
pk = Powerkit(feature_data, label_data)

# Register all conditions (method × size × model combinations)
rngs = np.random.RandomState(42).randint(0, 100000, size=1)  # 50 repeats for robustness

start_time = time.time()

for method_name, selection_method in feature_selection_methods.items():
    for k in feature_set_sizes:
        for model_name in models:
            # Create condition name using the requested naming convention
            condition = f"{method_name}_k{k}_{model_name}"
            
            # Create pipeline for this method and size
            pipeline_func = create_feature_selection_pipeline(selection_method, k, method_name, model_name)
            
            # Add condition to Powerkit following the structure from previous notebook [1]
            pk.add_condition(
                condition=condition,
                get_importance=True,
                pipeline_function=pipeline_func,
                pipeline_args={},
                eval_function=feature_selection_eval,
                eval_args={"metric_primary": "r2"}
            )

print(f"Registered {len(pk.conditions)} conditions in {time.time() - start_time:.2f} seconds")

# %%
# Run all conditions using Powerkit's parallel processing [1]
print("Starting feature selection benchmark (ANOVA-filter, MRMR, Mutual Information)...")
print(f"Running with {len(rngs)} random seeds and -1 n_jobs for maximum parallelization")

benchmark_start = time.time()
df_benchmark = pk.run_all_conditions(rng_list=rngs, n_jobs=-1, verbose=True)
benchmark_time = time.time() - benchmark_start

print(f"Benchmark completed in {benchmark_time:.2f} seconds")
print(f"Results shape: {df_benchmark.shape}")

# %%
# Extract k value and model name from condition for easier analysis
df_benchmark["k_value"] = df_benchmark["condition"].str.extract(r'k(\d+)').astype(int)
df_benchmark["method"] = df_benchmark["condition"].str.split('_').str[0]
df_benchmark["model_name"] = df_benchmark["condition"].str.split('_').str[2]

print("Condition breakdown:")
print(df_benchmark[["condition", "method", "k_value", "model_name"]].head(10))

# %%
# Save initial results
df_benchmark.to_pickle(f"{file_save_path}feature_selection_benchmark_{exp_id}.pkl")
print(f"Initial results saved to: {file_save_path}feature_selection_benchmark_{exp_id}.pkl")

# %%
# Quick summary of results
print("Benchmark Summary:")
print(f"Total runs: {len(df_benchmark)}")
print(f"Unique conditions: {df_benchmark['condition'].nunique()}")
print(f"Performance range (R²): {df_benchmark['model_performance'].min():.4f} to {df_benchmark['model_performance'].max():.4f}")

# Show performance by method (using the three requested methods)
method_summary = df_benchmark.groupby("method")["model_performance"].agg(["mean", "std", "count"])
print("\nPerformance by method (ANOVA-filter, MRMR, Mutual Information):")
print(method_summary.round(4))

# %% [markdown]
# ## Results and Visualisation

# %% [markdown]
# ### Heatmaps
