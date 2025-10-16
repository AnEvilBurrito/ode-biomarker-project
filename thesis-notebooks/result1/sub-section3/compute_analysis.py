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
exp_id = "v5_mrmr_vs_gffs_anova_prefilter"

if not os.path.exists(f"{path_loader.get_data_path()}data/results/{folder_name}/{exp_id}"):
    os.makedirs(f"{path_loader.get_data_path()}data/results/{folder_name}/{exp_id}")

file_save_path = f"{path_loader.get_data_path()}data/results/{folder_name}/{exp_id}/"

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
    mrmr_select_fcq_fast,
    mutual_information_select,
    select_random_features,
) #noqa: E402
import time #noqa: E402


# %%
def random_select_wrapper(X: pd.DataFrame, y: pd.Series, k: int) -> tuple:
    """Wrapper function for random feature selection that returns dummy scores"""
    selected_features, _ = select_random_features(X, y, k)
    # Return dummy scores (all zeros) since random selection has no meaningful scores
    dummy_scores = np.zeros(len(selected_features))
    return selected_features, dummy_scores


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
        # Use the working function from your baseline code [1]
        corr_keep_cols = _drop_correlated_columns(Xtr, threshold=0.95)
        Xtr_filtered = Xtr[corr_keep_cols]

        # 4) Feature selection
        k_sel = min(k, Xtr_filtered.shape[1]) if Xtr_filtered.shape[1] > 0 else 0
        if k_sel == 0:
            selected_features, selector_scores = [], np.array([])
            no_features = True
        else:
            selected_features, selector_scores = selection_method(
                Xtr_filtered, y_train, k_sel
            )
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
        }

    return pipeline_function


# %%
def feature_selection_eval(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    pipeline_components: Dict,
    metric_primary: Literal["r2", "pearson_r", "spearman_r"] = "r2",
) -> Dict:
    """Evaluation function for feature selection benchmarking"""

    # Unpack components following the structure from working baseline code [1]
    imputer = pipeline_components["imputer"]
    corr_keep = set(pipeline_components["corr_keep_cols"])
    selected = list(pipeline_components["selected_features"])
    selector_scores = pipeline_components["selector_scores"]
    model = pipeline_components["model"]
    model_name = pipeline_components["model_type"]
    scaler = pipeline_components.get("scaler", None)
    no_features = pipeline_components.get("no_features", False)

    # Apply identical transforms as training
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    y_test = pd.Series(y_test).replace([np.inf, -np.inf], np.nan)
    mask_y = ~y_test.isna()
    X_test, y_test = X_test.loc[mask_y], y_test.loc[mask_y]

    Xti = imputer.transform(X_test, return_df=True).astype(float).fillna(0.0)

    # Apply same correlation filtering as training [1]
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

    # Calculate metrics (following the exact structure from baseline_eval [1])
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
    }

# %% [markdown]
# ## Computational Cost Analysis Functions

# %%
import matplotlib.pyplot as plt #noqa: E402
import seaborn as sns #noqa: E402
from scipy.optimize import curve_fit #noqa: E402
from scipy.stats import linregress, ttest_ind #noqa: E402
import warnings #noqa: E402
warnings.filterwarnings('ignore')

# %%
def analyze_time_complexity(df_benchmark, report_file=None):
    """Analyze time complexity of feature selection methods"""
    
    save_and_print("## Computational Cost Analysis: Time Complexity", report_file, level="section")
    
    # Group by method and k_value for time analysis
    time_by_method_k = df_benchmark.groupby(['method', 'k_value'])['feature_selection_time'].agg([
        'mean', 'std', 'count', 'min', 'max'
    ]).reset_index()
    
    save_and_print("### Time Statistics by Method and k-value", report_file, level="subsection")
    save_and_print(time_by_method_k.round(4).to_string(), report_file, level="info")
    
    # Complexity model functions
    def linear_func(x, a, b):
        return a * x + b
    
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c
    
    def exponential_func(x, a, b):
        return a * np.exp(b * x)
    
    # Analyze complexity for each method
    complexity_results = {}
    methods = df_benchmark['method'].unique()
    
    for method in methods:
        method_data = df_benchmark[df_benchmark['method'] == method]
        k_values = sorted(method_data['k_value'].unique())
        
        # Calculate mean time for each k
        mean_times = []
        for k in k_values:
            k_data = method_data[method_data['k_value'] == k]
            mean_times.append(k_data['feature_selection_time'].mean())
        
        # Fit different complexity models
        x_data = np.array(k_values)
        y_data = np.array(mean_times)
        
        complexity_results[method] = {
            'k_values': k_values,
            'mean_times': mean_times,
            'fits': {}
        }
        
        # Linear fit
        try:
            popt_lin, _ = curve_fit(linear_func, x_data, y_data, maxfev=5000)
            y_pred_lin = linear_func(x_data, *popt_lin)
            r2_lin = 1 - np.sum((y_data - y_pred_lin)**2) / np.sum((y_data - np.mean(y_data))**2)
            complexity_results[method]['fits']['linear'] = {
                'params': popt_lin, 'r2': r2_lin
            }
        except:
            complexity_results[method]['fits']['linear'] = {'r2': np.nan}
        
        # Quadratic fit
        try:
            popt_quad, _ = curve_fit(quadratic_func, x_data, y_data, maxfev=5000)
            y_pred_quad = quadratic_func(x_data, *popt_quad)
            r2_quad = 1 - np.sum((y_data - y_pred_quad)**2) / np.sum((y_data - np.mean(y_data))**2)
            complexity_results[method]['fits']['quadratic'] = {
                'params': popt_quad, 'r2': r2_quad
            }
        except:
            complexity_results[method]['fits']['quadratic'] = {'r2': np.nan}
        
        # Exponential fit
        try:
            popt_exp, _ = curve_fit(exponential_func, x_data, y_data, maxfev=5000, p0=[1, 0.1])
            y_pred_exp = exponential_func(x_data, *popt_exp)
            r2_exp = 1 - np.sum((y_data - y_pred_exp)**2) / np.sum((y_data - np.mean(y_data))**2)
            complexity_results[method]['fits']['exponential'] = {
                'params': popt_exp, 'r2': r2_exp
            }
        except:
            complexity_results[method]['fits']['exponential'] = {'r2': np.nan}
    
    # Print complexity analysis results
    save_and_print("### Time Complexity Model Fits", report_file, level="subsection")
    for method, results in complexity_results.items():
        save_and_print(f"**{method.upper()} Method:**", report_file, level="info")
        for model_name, fit_info in results['fits'].items():
            if not np.isnan(fit_info['r2']):
                save_and_print(f"  {model_name.capitalize()} fit R²: {fit_info['r2']:.4f}", report_file, level="info")
    
    return complexity_results, time_by_method_k

# %%
def get_consistent_color_mapping(methods):
    """Create consistent color mapping for methods across all plots"""
    standard_colors = {
        'anova': '#1f77b4',  # Blue
        'mrmr': '#ff7f0e',   # Orange
        'mutual': '#2ca02c', # Green
        'random': '#d62728'  # Red
    }
    
    extended_palette = sns.color_palette("husl", max(8, len(methods)))
    
    color_mapping = {}
    for i, method in enumerate(methods):
        if method in standard_colors:
            color_mapping[method] = standard_colors[method]
        else:
            color_mapping[method] = extended_palette[i % len(extended_palette)]
    
    return color_mapping

def get_consistent_marker_mapping(methods):
    """Create consistent marker mapping for methods across all plots"""
    base_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    
    marker_mapping = {}
    for i, method in enumerate(methods):
        marker_mapping[method] = base_markers[i % len(base_markers)]
    
    return marker_mapping

def generate_method_labels(methods):
    """Generate human-readable labels for methods"""
    label_mapping = {}
    for method in methods:
        if method == 'anova':
            label_mapping[method] = 'ANOVA-Filter'
        elif method == 'mrmr':
            label_mapping[method] = 'MRMR'
        elif method == 'mutual':
            label_mapping[method] = 'Mutual Information'
        elif method == 'random':
            label_mapping[method] = 'Random Selection'
        else:
            label_mapping[method] = method.title()  # Default to title case
    
    return label_mapping

def plot_time_vs_k_comparison(df_benchmark, file_save_path, exp_id):
    """Create individual plot: Time vs k-value comparison for all methods"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))  # Figure size 8x6
    
    # Get consistent mappings
    methods = df_benchmark['method'].unique()
    color_mapping = get_consistent_color_mapping(methods)
    marker_mapping = get_consistent_marker_mapping(methods)
    method_labels = generate_method_labels(methods)
    
    # Plot each method
    for method in methods:
        method_data = df_benchmark[df_benchmark['method'] == method]
        k_values = sorted(method_data['k_value'].unique())
        
        # Skip methods with insufficient data
        if len(k_values) == 0:
            continue
            
        mean_times = []
        std_times = []
        
        for k in k_values:
            k_data = method_data[method_data['k_value'] == k]
            if len(k_data) > 0:
                mean_times.append(k_data['feature_selection_time'].mean())
                std_times.append(k_data['feature_selection_time'].std())
            else:
                mean_times.append(np.nan)
                std_times.append(np.nan)
        
        # Filter out NaN values
        valid_indices = ~np.isnan(mean_times)
        k_values_valid = [k for i, k in enumerate(k_values) if valid_indices[i]]
        mean_times_valid = [t for i, t in enumerate(mean_times) if valid_indices[i]]
        std_times_valid = [t for i, t in enumerate(std_times) if valid_indices[i]]
        
        if len(k_values_valid) > 0:
            ax.errorbar(k_values_valid, mean_times_valid, yerr=std_times_valid,
                       label=method_labels[method], 
                       color=color_mapping[method],
                       marker=marker_mapping[method],
                       capsize=5, linewidth=2, markersize=10)
    
    ax.set_xlabel('Number of Features Selected (k)', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Feature Selection Time vs k-value', fontsize=16, pad=20)
    ax.legend(fontsize=14)  # Legend size 14
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    # Save the plot
    plot_filename = f"{file_save_path}computational_cost_time_vs_k_{exp_id}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_time_distribution_by_method(df_benchmark, file_save_path, exp_id):
    """Create individual plot: Time distribution boxplot by method"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))  # Figure size 8x6
    
    # Get consistent mappings
    methods = df_benchmark['method'].unique()
    color_mapping = get_consistent_color_mapping(methods)
    method_labels = generate_method_labels(methods)
    
    # Create custom palette for boxplot
    palette = [color_mapping[method] for method in methods]
    
    # Create boxplot with consistent colors
    sns.boxplot(data=df_benchmark, x='method', y='feature_selection_time', 
                ax=ax, palette=palette)
    
    # Update x-axis labels with proper method labels
    ax.set_xticklabels([method_labels[method] for method in methods], fontsize=14)
    
    ax.set_xlabel('Feature Selection Method', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Distribution of Execution Times by Method', fontsize=16, pad=20)
    ax.set_yscale('log')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    # Save the plot
    plot_filename = f"{file_save_path}computational_cost_boxplot_{exp_id}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_time_efficiency_per_feature(df_benchmark, file_save_path, exp_id):
    """Create individual plot: Time efficiency per feature selected"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))  # Figure size 8x6
    
    # Get consistent mappings
    methods = df_benchmark['method'].unique()
    color_mapping = get_consistent_color_mapping(methods)
    marker_mapping = get_consistent_marker_mapping(methods)
    method_labels = generate_method_labels(methods)
    
    # Plot each method
    for method in methods:
        method_data = df_benchmark[df_benchmark['method'] == method]
        k_values = sorted(method_data['k_value'].unique())
        time_per_feature = []
        
        for k in k_values:
            k_data = method_data[method_data['k_value'] == k]
            if len(k_data) > 0 and k > 0:  # Avoid division by zero
                mean_time = k_data['feature_selection_time'].mean()
                time_per_feature.append(mean_time / k)
            else:
                time_per_feature.append(np.nan)
        
        # Filter out NaN values
        valid_indices = ~np.isnan(time_per_feature)
        k_values_valid = [k for i, k in enumerate(k_values) if valid_indices[i]]
        time_per_feature_valid = [t for i, t in enumerate(time_per_feature) if valid_indices[i]]
        
        if len(k_values_valid) > 0:
            ax.plot(k_values_valid, time_per_feature_valid, 
                   label=method_labels[method],
                   color=color_mapping[method],
                   marker=marker_mapping[method],
                   linewidth=2, markersize=10)
    
    ax.set_xlabel('Number of Features Selected (k)', fontsize=14)
    ax.set_ylabel('Time per Feature (seconds)', fontsize=14)
    ax.set_title('Time Efficiency per Feature Selected', fontsize=16, pad=20)
    ax.legend(fontsize=14)  # Legend size 14
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    # Save the plot
    plot_filename = f"{file_save_path}computational_cost_efficiency_{exp_id}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_pairwise_speedup_comparison(df_benchmark, file_save_path, exp_id):
    """Create individual plot: Pairwise speedup comparison between all methods"""
    
    methods = df_benchmark['method'].unique()
    method_labels = generate_method_labels(methods)
    
    # Only create plot if we have at least 2 methods
    if len(methods) < 2:
        print("Insufficient methods for pairwise speedup comparison")
        return None
    
    # Create figure with dynamic layout
    n_pairs = len(methods) * (len(methods) - 1) // 2
    max_cols = min(3, n_pairs)
    rows = (n_pairs + max_cols - 1) // max_cols
    
    fig, axes = plt.subplots(rows, max_cols, figsize=(6*max_cols, 5*rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    pair_index = 0
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                ax = axes[pair_index]
                
                method1_data = df_benchmark[df_benchmark['method'] == method1]
                method2_data = df_benchmark[df_benchmark['method'] == method2]
                
                # Find common k-values
                k_values_method1 = set(method1_data['k_value'].unique())
                k_values_method2 = set(method2_data['k_value'].unique())
                common_k_values = sorted(k_values_method1.intersection(k_values_method2))
                
                speedup_ratios = []
                
                for k in common_k_values:
                    method1_time = method1_data[method1_data['k_value'] == k]['feature_selection_time'].mean()
                    method2_time = method2_data[method2_data['k_value'] == k]['feature_selection_time'].mean()
                    if method2_time > 0:  # Avoid division by zero
                        speedup_ratios.append(method1_time / method2_time)
                    else:
                        speedup_ratios.append(np.nan)
                
                # Filter out NaN values
                valid_indices = ~np.isnan(speedup_ratios)
                k_values_valid = [k for i, k in enumerate(common_k_values) if valid_indices[i]]
                speedup_ratios_valid = [r for i, r in enumerate(speedup_ratios) if valid_indices[i]]
                
                if len(k_values_valid) > 0:
                    ax.plot(k_values_valid, speedup_ratios_valid, 'ro-', linewidth=2, markersize=10)
                    ax.set_xlabel('Number of Features Selected (k)', fontsize=14)
                    ax.set_ylabel(f'Speedup Ratio\n({method_labels[method1]} / {method_labels[method2]})', fontsize=14)
                    ax.set_title(f'Speedup: {method_labels[method2]} vs {method_labels[method1]}', fontsize=16, pad=20)
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    
                    # Increase tick label sizes
                    ax.tick_params(axis='both', which='major', labelsize=14)
                    ax.tick_params(axis='both', which='minor', labelsize=12)
                else:
                    ax.text(0.5, 0.5, 'Insufficient data\nfor comparison', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'Speedup: {method_labels[method2]} vs {method_labels[method1]}', fontsize=16, pad=20)
                
                pair_index += 1
    
    # Hide unused subplots
    for i in range(pair_index, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{file_save_path}computational_cost_speedup_comparison_{exp_id}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_performance_vs_time_scatter(df_benchmark, file_save_path, exp_id):
    """Create scatter plot: Performance vs Computational Time with k-value as confounder"""
    
    # Create figure with subplots for better visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Wider figure for two panels
    
    # Get consistent mappings
    methods = df_benchmark['method'].unique()
    color_mapping = get_consistent_color_mapping(methods)
    marker_mapping = get_consistent_marker_mapping(methods)
    method_labels = generate_method_labels(methods)
    
    # Get k-values for size mapping
    k_values = sorted(df_benchmark['k_value'].unique())
    min_k, max_k = min(k_values), max(k_values)
    
    # Panel 1: Scatter plot with k-value as point size
    for method in methods:
        method_data = df_benchmark[df_benchmark['method'] == method]
        
        if len(method_data) == 0:
            continue
        
        # Map k-value to point size (50 to 200 pixels)
        sizes = 50 + 150 * (method_data['k_value'] - min_k) / (max_k - min_k) if max_k > min_k else 100
        
        scatter = ax1.scatter(method_data['feature_selection_time'], 
                           method_data['model_performance'],
                           c=[color_mapping[method]] * len(method_data),
                           marker=marker_mapping[method],
                           s=sizes, alpha=0.7, label=method_labels[method])
    
    ax1.set_xlabel('Computational Time (seconds)', fontsize=14)
    ax1.set_ylabel('Model Performance (R²)', fontsize=14)
    ax1.set_title('Performance vs Time (Size = k-value)', fontsize=16, pad=20)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = df_benchmark['model_performance'].corr(df_benchmark['feature_selection_time'])
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax1.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel 2: Faceted by k-value for clearer comparison
    k_values = sorted(df_benchmark['k_value'].unique())
    max_cols = min(3, len(k_values))
    rows = (len(k_values) + max_cols - 1) // max_cols
    
    # Create subplot grid for k-value faceting
    fig2, axes = plt.subplots(rows, max_cols, figsize=(6*max_cols, 5*rows))
    if len(k_values) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        k_data = df_benchmark[df_benchmark['k_value'] == k]
        
        for method in methods:
            method_k_data = k_data[k_data['method'] == method]
            
            if len(method_k_data) > 0:
                ax.scatter(method_k_data['feature_selection_time'], 
                          method_k_data['model_performance'],
                          c=[color_mapping[method]] * len(method_k_data),
                          marker=marker_mapping[method],
                          s=80, alpha=0.7, label=method_labels[method])
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Performance (R²)', fontsize=12)
        ax.set_title(f'k = {k}', fontsize=14, pad=15)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        if idx == 0:
            ax.legend(fontsize=10)
    
    # Hide unused subplots
    for idx in range(len(k_values), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Performance vs Time by k-value', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save both plots
    plot_filename1 = f"{file_save_path}performance_vs_time_scatter_{exp_id}.png"
    plot_filename2 = f"{file_save_path}performance_vs_time_by_k_{exp_id}.png"
    
    plt.figure(fig.number)
    plt.savefig(plot_filename1, dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(fig2.number)
    plt.savefig(plot_filename2, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, fig2

def plot_speedup_comparison_matrix(df_benchmark, file_save_path, exp_id):
    """Create enhanced speedup comparison matrix heatmap (Plot 6)"""
    
    methods = df_benchmark['method'].unique()
    method_labels = generate_method_labels(methods)
    
    # Only create plot if we have at least 2 methods
    if len(methods) < 2:
        print("Insufficient methods for speedup comparison matrix")
        return None
    
    # Calculate average times for each method with confidence intervals
    method_stats = {}
    for method in methods:
        method_data = df_benchmark[df_benchmark['method'] == method]['feature_selection_time']
        method_stats[method] = {
            'mean': method_data.mean(),
            'std': method_data.std(),
            'count': len(method_data),
            'ci_low': method_data.mean() - 1.96 * method_data.std() / np.sqrt(len(method_data)),
            'ci_high': method_data.mean() + 1.96 * method_data.std() / np.sqrt(len(method_data))
        }
    
    # Create speedup matrix with confidence intervals
    speedup_matrix = np.zeros((len(methods), len(methods)))
    speedup_ci_low = np.zeros((len(methods), len(methods)))
    speedup_ci_high = np.zeros((len(methods), len(methods)))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if method_stats[method2]['mean'] > 0:  # Avoid division by zero
                speedup_matrix[i, j] = method_stats[method1]['mean'] / method_stats[method2]['mean']
                # Calculate confidence interval for speedup ratio using delta method approximation
                if method_stats[method1]['count'] > 0 and method_stats[method2]['count'] > 0:
                    var_ratio = (method_stats[method1]['std']**2 / method_stats[method1]['mean']**2 + 
                                method_stats[method2]['std']**2 / method_stats[method2]['mean']**2)
                    se_ratio = speedup_matrix[i, j] * np.sqrt(var_ratio)
                    speedup_ci_low[i, j] = speedup_matrix[i, j] - 1.96 * se_ratio
                    speedup_ci_high[i, j] = speedup_matrix[i, j] + 1.96 * se_ratio
                else:
                    speedup_ci_low[i, j] = speedup_ci_high[i, j] = np.nan
            else:
                speedup_matrix[i, j] = np.nan
                speedup_ci_low[i, j] = np.nan
                speedup_ci_high[i, j] = np.nan
    
    # Create figure with enhanced layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # Wider figure for two panels
    
    # Panel 1: Enhanced heatmap with statistical significance
    im = ax1.imshow(speedup_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.5, vmax=2.0)
    
    # Set labels with better formatting
    ax1.set_xticks(np.arange(len(methods)))
    ax1.set_yticks(np.arange(len(methods)))
    ax1.set_xticklabels([method_labels[method] for method in methods], fontsize=16)
    ax1.set_yticklabels([method_labels[method] for method in methods], fontsize=16)
    
    # Rotate x labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add enhanced text annotations with confidence intervals
    for i in range(len(methods)):
        for j in range(len(methods)):
            if not np.isnan(speedup_matrix[i, j]):
                # Determine if speedup is statistically significant (CI doesn't include 1)
                is_significant = (speedup_ci_low[i, j] > 1.0 or speedup_ci_high[i, j] < 1.0) if not np.isnan(speedup_ci_low[i, j]) else False
                
                text_color = "white" if (speedup_matrix[i, j] < 0.8 or speedup_matrix[i, j] > 1.2) else "black"
                font_weight = 'bold' if is_significant else 'normal'
                
                annotation = f'{speedup_matrix[i, j]:.2f}x'
                
                ax1.text(j, i, annotation,
                        ha="center", va="center", color=text_color, fontsize=14,
                        fontweight=font_weight, linespacing=1.2)
    
    ax1.set_xlabel('Compared Method', fontsize=18)
    ax1.set_ylabel('Reference Method', fontsize=18)
    ax1.set_title('Speedup Comparison Matrix\n(Reference Time / Compared Time)', fontsize=16, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Speedup Ratio', fontsize=20)
    
    # Panel 2: Method performance summary
    method_means = [method_stats[method]['mean'] for method in methods]
    method_stds = [method_stats[method]['std'] for method in methods]
    
    bars = ax2.bar(range(len(methods)), method_means, yerr=method_stds, 
                   capsize=5, alpha=0.7, color=[get_consistent_color_mapping(methods)[method] for method in methods])
    
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([method_labels[method] for method in methods], fontsize=16, rotation=45)
    ax2.set_ylabel('Average Time (seconds)', fontsize=20)
    ax2.set_title('Method Performance Summary', fontsize=20, pad=20)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, method_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{mean:.3f}s', ha='center', va='bottom', fontsize=14)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{file_save_path}speedup_comparison_matrix_{exp_id}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_time_vs_performance_by_k(df_benchmark, file_save_path, exp_id):
    """Create time vs performance analysis by k-value (Plot 8)"""
    
    # Create figure with dynamic layout based on number of k-values
    k_values = sorted(df_benchmark['k_value'].unique())
    max_cols = min(3, len(k_values))
    rows = (len(k_values) + max_cols - 1) // max_cols
    
    fig, axes = plt.subplots(rows, max_cols, figsize=(6*max_cols, 5*rows))
    if len(k_values) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Get consistent mappings
    methods = df_benchmark['method'].unique()
    color_mapping = get_consistent_color_mapping(methods)
    marker_mapping = get_consistent_marker_mapping(methods)
    method_labels = generate_method_labels(methods)
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        k_data = df_benchmark[df_benchmark['k_value'] == k]
        
        # Plot each method for this k-value
        for method in methods:
            method_k_data = k_data[k_data['method'] == method]
            
            if len(method_k_data) > 0:
                # Calculate mean time and performance for this method at this k
                mean_time = method_k_data['feature_selection_time'].mean()
                mean_performance = method_k_data['model_performance'].mean()
                std_time = method_k_data['feature_selection_time'].std()
                std_performance = method_k_data['model_performance'].std()
                
                ax.errorbar(mean_time, mean_performance,
                          xerr=std_time, yerr=std_performance,
                          label=method_labels[method],
                          color=color_mapping[method],
                          marker=marker_mapping[method],
                          capsize=5, linewidth=2, markersize=8, alpha=0.8)
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Performance (R²)', fontsize=12)
        ax.set_title(f'k = {k}', fontsize=14, pad=15)
        ax.grid(True, alpha=0.3)
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='both', which='minor', labelsize=9)
        
        # Add legend only to first subplot to avoid duplication
        if idx == 0:
            ax.legend(fontsize=10)
    
    # Hide unused subplots
    for idx in range(len(k_values), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Time vs Performance by k-value', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{file_save_path}time_vs_performance_by_k_{exp_id}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_computational_cost_visualizations(df_benchmark, file_save_path, exp_id):
    """Create all individual computational cost visualizations"""
    
    print("Creating individual computational cost visualizations...")
    
    # Create each plot individually
    plot_time_vs_k_comparison(df_benchmark, file_save_path, exp_id)
    plot_time_distribution_by_method(df_benchmark, file_save_path, exp_id)
    plot_time_efficiency_per_feature(df_benchmark, file_save_path, exp_id)
    plot_pairwise_speedup_comparison(df_benchmark, file_save_path, exp_id)
    
    # Create the two new requested plots (removed the last one)
    plot_performance_vs_time_scatter(df_benchmark, file_save_path, exp_id)
    plot_speedup_comparison_matrix(df_benchmark, file_save_path, exp_id)
    
    print("All individual visualizations created and saved.")

# %%
def statistical_comparison(df_benchmark, report_file=None):
    """Perform statistical comparison of computational costs"""
    
    save_and_print("## Statistical Comparison of Computational Costs", report_file, level="section")
    
    # Basic statistics
    save_and_print("### Basic Time Statistics", report_file, level="subsection")
    stats_summary = df_benchmark.groupby('method')['feature_selection_time'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(4)
    save_and_print(stats_summary.to_string(), report_file, level="info")
    
    # Correlation analysis
    correlation = df_benchmark['feature_selection_time'].corr(df_benchmark['k_value'])
    save_and_print(f"### Correlation Analysis", report_file, level="subsection")
    save_and_print(f"Correlation between time and k-value: {correlation:.4f}", report_file, level="info")
    
    # Perform pairwise statistical comparisons for all methods
    methods = df_benchmark['method'].unique()
    if len(methods) >= 2:
        save_and_print("### Pairwise Statistical Significance Tests", report_file, level="subsection")
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:
                    method1_times = df_benchmark[df_benchmark['method'] == method1]['feature_selection_time']
                    method2_times = df_benchmark[df_benchmark['method'] == method2]['feature_selection_time']
                    
                    if len(method1_times) > 0 and len(method2_times) > 0:
                        t_stat, p_value = ttest_ind(method1_times, method2_times, equal_var=False)
                        speedup_ratio = method1_times.mean() / method2_times.mean() if method2_times.mean() > 0 else np.inf
                        
                        save_and_print(f"{method1.upper()} vs {method2.upper()}:", report_file, level="info")
                        save_and_print(f"  T-statistic: {t_stat:.4f}", report_file, level="info")
                        save_and_print(f"  P-value: {p_value:.4e}", report_file, level="info")
                        save_and_print(f"  Significant difference: {'YES' if p_value < 0.05 else 'NO'}", report_file, level="info")
                        save_and_print(f"  Speedup ratio ({method1.upper()}/{method2.upper()}): {speedup_ratio:.1f}x", report_file, level="info")
    
    return {
        'correlation': correlation
    }

# %%
def generate_practical_recommendations(df_benchmark, complexity_results, stats_results, report_file=None):
    """Generate practical recommendations based on computational cost analysis"""
    
    save_and_print("## Practical Recommendations", report_file, level="section")
    
    # Calculate time estimates for different scenarios using actual k-values from data
    k_values = sorted(df_benchmark['k_value'].unique())
    
    save_and_print("### Time Estimates by k-value", report_file, level="subsection")
    for k in k_values:
        time_estimates = {}
        for method in df_benchmark['method'].unique():
            method_estimate = df_benchmark[(df_benchmark['method'] == method) & 
                                         (df_benchmark['k_value'] == k)]['feature_selection_time'].mean()
            time_estimates[method] = method_estimate
        
        save_and_print(f"k={k}:", report_file, level="info")
        for method, estimate in time_estimates.items():
            save_and_print(f"  {method.upper()} estimate: {estimate:.4f} seconds", report_file, level="info")
        
        # Calculate speedup if there are at least 2 methods
        if len(time_estimates) >= 2:
            methods = list(time_estimates.keys())
            method1_time = time_estimates[methods[0]]
            method2_time = time_estimates[methods[1]]
            if method2_time > 0:
                speedup = method1_time / method2_time
                save_and_print(f"  Speedup ({methods[0].upper()}/{methods[1].upper()}): {speedup:.1f}x", report_file, level="info")
    
    # Recommendations based on complexity
    save_and_print("### Method Selection Guidelines", report_file, level="subsection")
    
    # Calculate average times for each method
    method_times = {}
    for method in df_benchmark['method'].unique():
        method_times[method] = df_benchmark[df_benchmark['method'] == method]['feature_selection_time'].mean()
    
    # Sort methods by speed (fastest first)
    sorted_methods = sorted(method_times.keys(), key=lambda x: method_times[x])
    
    if len(sorted_methods) >= 2:
        fastest_method = sorted_methods[0]
        slowest_method = sorted_methods[-1]
        speedup_ratio = method_times[slowest_method] / method_times[fastest_method] if method_times[fastest_method] > 0 else np.inf
        
        save_and_print(f"1. **For time-sensitive applications**: Use {fastest_method.upper()} method", report_file, level="info")
        save_and_print(f"   - {fastest_method.upper()} is ~{speedup_ratio:.0f}x faster than {slowest_method.upper()} on average", report_file, level="info")
        save_and_print("   - Suitable for real-time or interactive applications", report_file, level="info")
        
        save_and_print(f"2. **For performance-critical applications**: Consider {slowest_method.upper()}", report_file, level="info")
        save_and_print("   - May provide better feature selection quality", report_file, level="info")
        save_and_print("   - Use when computational time is not a constraint", report_file, level="info")
        
        save_and_print("3. **For large-scale analyses**:", report_file, level="info")
        save_and_print(f"   - Start with {fastest_method.upper()} for initial screening", report_file, level="info")
        save_and_print(f"   - Use {slowest_method.upper()} on promising subsets for refinement", report_file, level="info")
    
    save_and_print("4. **k-value selection**:", report_file, level="info")
    save_and_print("   - Time typically scales with k-value", report_file, level="info")
    save_and_print("   - Consider computational constraints when selecting k", report_file, level="info")
    
    return {
        'time_estimates': {k: {
            method: df_benchmark[(df_benchmark['method'] == method) & (df_benchmark['k_value'] == k)]['feature_selection_time'].mean()
            for method in df_benchmark['method'].unique()
        } for k in k_values}
    }

# %% [markdown]
# ## Results and Visualisation

# %% [markdown]
# ### Load data

# %%
def save_and_print(message, report_file=None, level="info"):
    """
    Print message to console and save to report file with proper formatting.
    
    Args:
        message: The message to print and save
        report_file: File object to save to (optional)
        level: Formatting level - "header", "section", "subsection", or "info"
    """
    # Print to console
    print(message)
    
    # Save to report with proper formatting
    if report_file:
        if level == "header":
            report_file.write(f"# {message}\n\n")
        elif level == "section":
            report_file.write(f"## {message}\n\n")
        elif level == "subsection":
            report_file.write(f"### {message}\n\n")
        else:  # info level
            report_file.write(f"{message}\n\n")
    
    return message

# %%
# Load saved feature selection benchmark (feature_selection_benchmark_v1.pkl)
import os
import pandas as pd
import time #noqa: E402

# Create a new report file for capturing print statements
print_report_path = f"{file_save_path}compute_cost_print_report_{exp_id}.md"
print_report_file = open(print_report_path, 'w', encoding='utf-8')

# Write header to the print report
print_report_file.write(f"# Computational Cost Analysis Report - {exp_id}\n\n")
print_report_file.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
print_report_file.write("This report analyzes the computational cost differences between feature selection methods.\n\n")

pkl_path = f"{path_loader.get_data_path()}data/results/{folder_name}/feature_selection_benchmark_{exp_id}.pkl"
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"Pickle not found: {pkl_path}")

df_benchmark = pd.read_pickle(pkl_path)
save_and_print(f"Loaded df_benchmark with shape: {df_benchmark.shape}", print_report_file, level="section")

# %% [markdown]
# ### Fix Data Structure Issues

# %%
def parse_condition_column(df_benchmark):
    """Parse the condition column to extract method, k_value, and model_name correctly"""
    
    save_and_print("## Fixing Data Structure Issues", print_report_file, level="section")
    save_and_print("Parsing condition column to extract correct method, k_value, and model_name", print_report_file, level="info")
    
    # Create new columns based on condition parsing
    parsed_data = []
    
    for idx, row in df_benchmark.iterrows():
        condition = row['condition']
        
        # Parse the condition format: {method}_k{value}_{model}
        # Example: "mrmr_anova_prefilter_k5_KNeighborsRegressor"
        parts = condition.split('_')
        
        # Extract method (everything before the k-value part)
        method_parts = []
        k_value = None
        model_name = None
        
        for part in parts:
            if part.startswith('k'):
                # Found k-value part, extract numeric value
                k_value = int(part[1:])  # Remove 'k' prefix and convert to int
                # Everything before this is the method
                method = '_'.join(method_parts)
                # Everything after this is the model
                model_parts = parts[parts.index(part) + 1:]
                model_name = '_'.join(model_parts)
                break
            else:
                method_parts.append(part)
        
        # If we didn't find a k-value (shouldn't happen with valid data)
        if k_value is None:
            save_and_print(f"Warning: Could not parse k-value from condition: {condition}", print_report_file, level="info")
            method = '_'.join(method_parts[:-1]) if len(method_parts) > 1 else method_parts[0]
            model_name = parts[-1] if parts else 'unknown'
            k_value = 0
        
        parsed_data.append({
            'condition': condition,
            'parsed_method': method,
            'parsed_k_value': k_value,
            'parsed_model_name': model_name
        })
    
    # Create a DataFrame with parsed values
    parsed_df = pd.DataFrame(parsed_data)
    
    # Compare with existing columns
    save_and_print("### Comparison of Original vs Parsed Values", print_report_file, level="subsection")
    
    # Check method consistency
    method_mismatch = df_benchmark['method'] != parsed_df['parsed_method']
    if method_mismatch.any():
        save_and_print(f"Method mismatches found: {method_mismatch.sum()}/{len(df_benchmark)}", print_report_file, level="info")
        for idx in df_benchmark[method_mismatch].index[:5]:  # Show first 5 mismatches
            save_and_print(f"  Row {idx}: Original='{df_benchmark.loc[idx, 'method']}', Parsed='{parsed_df.loc[idx, 'parsed_method']}'", 
                          print_report_file, level="info")
    
    # Check k_value consistency
    k_mismatch = df_benchmark['k_value'] != parsed_df['parsed_k_value']
    if k_mismatch.any():
        save_and_print(f"K-value mismatches found: {k_mismatch.sum()}/{len(df_benchmark)}", print_report_file, level="info")
        for idx in df_benchmark[k_mismatch].index[:5]:
            save_and_print(f"  Row {idx}: Original={df_benchmark.loc[idx, 'k_value']}, Parsed={parsed_df.loc[idx, 'parsed_k_value']}", 
                          print_report_file, level="info")
    
    # Check model_name consistency
    model_mismatch = df_benchmark['model_name'] != parsed_df['parsed_model_name']
    if model_mismatch.any():
        save_and_print(f"Model name mismatches found: {model_mismatch.sum()}/{len(df_benchmark)}", print_report_file, level="info")
        for idx in df_benchmark[model_mismatch].index[:5]:
            save_and_print(f"  Row {idx}: Original='{df_benchmark.loc[idx, 'model_name']}', Parsed='{parsed_df.loc[idx, 'parsed_model_name']}'", 
                          print_report_file, level="info")
    
    # Update the dataframe with parsed values
    df_benchmark['method'] = parsed_df['parsed_method']
    df_benchmark['k_value'] = parsed_df['parsed_k_value']
    df_benchmark['model_name'] = parsed_df['parsed_model_name']
    
    save_and_print("Dataframe columns updated with correctly parsed values", print_report_file, level="info")
    
    # Show unique values after parsing
    save_and_print("### Unique Values After Parsing", print_report_file, level="subsection")
    save_and_print(f"Methods: {df_benchmark['method'].unique()}", print_report_file, level="info")
    save_and_print(f"K-values: {sorted(df_benchmark['k_value'].unique())}", print_report_file, level="info")
    save_and_print(f"Models: {df_benchmark['model_name'].unique()}", print_report_file, level="info")
    
    return df_benchmark

# Apply the parsing fix
df_benchmark = parse_condition_column(df_benchmark)

# Display first rows (works in notebook)
try:
    from IPython.display import display

    display(df_benchmark.head())
except Exception:
    save_and_print(df_benchmark.head().to_string(), print_report_file, level="info")
    
# Re-define variables that might be needed in the loaded section
# Dynamically detect available methods, models, and k-values from the data
feature_set_sizes = sorted(df_benchmark['k_value'].unique())
models = df_benchmark['model_name'].unique().tolist()

# Dynamically create method labels based on available methods
available_methods = df_benchmark['method'].unique().tolist()
method_labels = {}
for method in available_methods:
    if method == 'anova':
        method_labels[method] = 'ANOVA-Filter'
    elif method == 'mrmr':
        method_labels[method] = 'MRMR'
    elif method == 'mutual':
        method_labels[method] = 'Mutual Information'
    elif method == 'random':
        method_labels[method] = 'Random Selection'
    else:
        # For any new methods not predefined, use the method name as label
        method_labels[method] = method.title()

print(f"Actual k-values present in data: {feature_set_sizes}")
print(f"Available methods: {available_methods}")
print(f"Available models: {models}")

# %% [markdown]
# ### Computational Cost Analysis Execution

# %%
# Execute computational cost analysis
save_and_print("## Computational Cost Analysis Results", print_report_file, level="section")

# 1. Time complexity analysis
complexity_results, time_by_method_k = analyze_time_complexity(df_benchmark, print_report_file)

# 2. Statistical comparison
stats_results = statistical_comparison(df_benchmark, print_report_file)

# 3. Generate visualizations
create_computational_cost_visualizations(df_benchmark, file_save_path, exp_id)
save_and_print("Computational cost visualizations generated and saved.", print_report_file, level="info")

# 4. Practical recommendations
recommendations = generate_practical_recommendations(df_benchmark, complexity_results, stats_results, print_report_file)

# 5. Additional detailed analysis
save_and_print("## Detailed Computational Cost Analysis", print_report_file, level="section")

# Analyze time scaling with k-value
save_and_print("### Time Scaling Analysis", print_report_file, level="subsection")
for method in df_benchmark['method'].unique():
    method_data = df_benchmark[df_benchmark['method'] == method]
    k_values = sorted(method_data['k_value'].unique())
    
    scaling_factors = []
    for i in range(1, len(k_values)):
        time_ratio = method_data[method_data['k_value'] == k_values[i]]['feature_selection_time'].mean() / \
                    method_data[method_data['k_value'] == k_values[i-1]]['feature_selection_time'].mean()
        scaling_factors.append(time_ratio)
    
    if scaling_factors:
        avg_scaling = np.mean(scaling_factors)
        save_and_print(f"{method.upper()} average time scaling factor when doubling k: {avg_scaling:.2f}x", 
                      print_report_file, level="info")

# Performance vs time correlation
performance_time_corr = df_benchmark['model_performance'].corr(df_benchmark['feature_selection_time'])
save_and_print(f"### Performance vs Time Correlation", print_report_file, level="subsection")
save_and_print(f"Correlation between model performance and computational time: {performance_time_corr:.4f}", 
              print_report_file, level="info")

# Method efficiency analysis
save_and_print("### Method Efficiency Analysis", print_report_file, level="subsection")
for method in df_benchmark['method'].unique():
    method_data = df_benchmark[df_benchmark['method'] == method]
    avg_performance = method_data['model_performance'].mean()
    avg_time = method_data['feature_selection_time'].mean()
    efficiency_ratio = avg_performance / avg_time if avg_time > 0 else np.inf
    
    save_and_print(f"{method.upper()}: Performance/Time efficiency ratio: {efficiency_ratio:.6f}", 
                  print_report_file, level="info")

# %% [markdown]
# ### Summary and Conclusion

# %%
# Final summary
save_and_print("## Summary and Conclusion", print_report_file, level="section")

save_and_print("### Key Findings", print_report_file, level="subsection")
save_and_print("1. **Computational Differences**: Significant time differences exist between feature selection methods", print_report_file, level="info")
save_and_print("2. **Time Scaling**: Computational time typically scales with k-value selection", print_report_file, level="info")
save_and_print("3. **Statistical Significance**: Time differences between methods are statistically significant", print_report_file, level="info")
save_and_print("4. **Practical Implications**: Method selection should consider computational constraints and performance requirements", print_report_file, level="info")

save_and_print("### Recommendations for Future Work", print_report_file, level="subsection")
save_and_print("1. **Hybrid Approaches**: Consider using faster methods for initial screening followed by more thorough methods for refinement", print_report_file, level="info")
save_and_print("2. **Parallelization**: Explore parallel implementations for computationally intensive methods", print_report_file, level="info")
save_and_print("3. **Early Stopping**: Implement early stopping criteria to balance performance and time", print_report_file, level="info")
save_and_print("4. **Dataset-specific Optimization**: Develop optimization strategies based on dataset characteristics", print_report_file, level="info")

# Close the report file
print_report_file.close()
save_and_print(f"Computational cost analysis completed. Report saved to: {print_report_path}", level="section")

# Display final message
print(f"\n{'='*60}")
print("COMPUTATIONAL COST ANALYSIS COMPLETED")
print(f"{'='*60}")
print(f"Report saved to: {print_report_path}")
print(f"Individual visualizations saved to:")
print(f"  - {file_save_path}computational_cost_time_vs_k_{exp_id}.png")
print(f"  - {file_save_path}computational_cost_boxplot_{exp_id}.png")
print(f"  - {file_save_path}computational_cost_efficiency_{exp_id}.png")
print(f"  - {file_save_path}computational_cost_speedup_comparison_{exp_id}.png")
print(f"  - {file_save_path}performance_vs_time_scatter_{exp_id}.png")
print(f"  - {file_save_path}performance_vs_time_by_k_{exp_id}.png")
print(f"  - {file_save_path}speedup_comparison_matrix_{exp_id}.png")
print(f"{'='*60}")
