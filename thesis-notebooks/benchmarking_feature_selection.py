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
# ## Execution 
# %%
# Setup experiment parameters - using only the three requested methods
feature_set_sizes = [10, 20, 40, 80, 160, 320, 640, 1280]
models = ["KNeighborsRegressor", "LinearRegression", "SVR"]

# Define the feature selection methods including random selection as negative control
feature_selection_methods = {
    "anova_filter": f_regression_select,        # Renamed as ANOVA-filter
    "mrmr": mrmr_select_fcq_fast,                    # MRMR method [2]
    "mutual_info": mutual_information_select,   # Mutual Information method [2]
    "random_select": random_select_wrapper      # Random selection as negative control
}

print(f"Benchmarking {len(feature_selection_methods)} methods across {len(feature_set_sizes)} feature sizes and {len(models)} models")
print(f"Total conditions: {len(feature_selection_methods) * len(feature_set_sizes) * len(models)}")
print("Methods: ANOVA-filter, MRMR, Mutual Information, and Random Selection (negative control)")

# %%
from toolkit import Powerkit #noqa: E402
import numpy as np #noqa: E402
import time #noqa: F811, E402

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
print("Starting feature selection benchmark (ANOVA-filter, MRMR, Mutual Information, Random Selection)...")
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

# Show performance by method (including random selection as negative control)
method_summary = df_benchmark.groupby("method")["model_performance"].agg(["mean", "std", "count"])
print("\nPerformance by method (ANOVA-filter, MRMR, Mutual Information, Random Selection):")
print(method_summary.round(4))

# %% [markdown]
# ## Results and Visualisation

# %% [markdown]
# ### Load data

# %%
# Load saved feature selection benchmark (feature_selection_benchmark_v1.pkl)
import os
import pandas as pd

pkl_path = f"{path_loader.get_data_path()}data/results/{folder_name}/feature_selection_benchmark_{exp_id}.pkl"
if not os.path.exists(pkl_path):
    raise FileNotFoundError(f"Pickle not found: {pkl_path}")

df_benchmark = pd.read_pickle(pkl_path)
print(f"Loaded df_benchmark with shape: {df_benchmark.shape}")
# Display first rows (works in notebook)
try:
    from IPython.display import display

    display(df_benchmark.head())
except Exception:
    print(df_benchmark.head().to_string())


# %%
# Basic stats


# %% [markdown]
# ### Performance Comparison: Feature Selection Methods

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Method labels for better visualization
method_labels = {
    'anova_filter': 'ANOVA-Filter',
    'mrmr': 'MRMR', 
    'mutual_info': 'Mutual Information',
    'random_select': 'Random Selection'
}

# Create publication-quality box plot comparing methods across all feature sizes and models
plt.figure(figsize=(10, 6), dpi=300)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# Define publication-quality color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

sns.boxplot(data=df_benchmark, x='method', y='model_performance', 
            order=['anova_filter', 'mrmr', 'mutual_info', 'random_select'],
            palette=colors, width=0.6, fliersize=3)
plt.title('Feature Selection Method Performance Comparison', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Feature Selection Method', fontsize=14, fontweight='bold')
plt.ylabel('R² Score', fontsize=14, fontweight='bold')
plt.xticks(ticks=range(4), 
           labels=[method_labels[m] for m in ['anova_filter', 'mrmr', 'mutual_info', 'random_select']],
           rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', alpha=0.2, linestyle='--')
plt.tight_layout()
plt.savefig(f"{file_save_path}method_comparison_boxplot.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Calculate mean and standard deviation for each method
method_stats = df_benchmark.groupby('method')['model_performance'].agg(['mean', 'std', 'count']).reset_index()

# Create publication-quality bar plot with error bars
plt.figure(figsize=(10, 6), dpi=300)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# Define publication-quality color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

# Create bar plot with error bars
bars = plt.bar(range(len(method_stats)), method_stats['mean'], 
               yerr=method_stats['std'], capsize=8, alpha=0.8,
               color=colors, edgecolor='black', linewidth=1)

plt.title('Mean Performance of Feature Selection Methods', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Feature Selection Method', fontsize=14, fontweight='bold')
plt.ylabel('Mean R² Score ± Std. Dev.', fontsize=14, fontweight='bold')
plt.xticks(ticks=range(len(method_stats)), 
           labels=[method_labels.get(m, m) for m in method_stats['method']],
           rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Add value labels on bars with improved formatting
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f} ± {method_stats.iloc[i]["std"]:.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(axis='y', alpha=0.2, linestyle='--')
plt.tight_layout()
plt.savefig(f"{file_save_path}method_performance_bar.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Create publication-quality faceted box plots by model type
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# Define publication-quality color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

g = sns.catplot(data=df_benchmark, x='method', y='model_performance', 
                col='model_name', kind='box', height=6, aspect=1.2,
                order=['anova_filter', 'mrmr', 'mutual_info', 'random_select'],
                palette=colors, width=0.6, fliersize=3)
g.set_titles("Model: {col_name}", fontsize=14, fontweight='bold')
g.set_axis_labels("Feature Selection Method", "R² Score", fontsize=12, fontweight='bold')
g.fig.suptitle('Feature Selection Performance by ML Model', y=1.02, fontsize=16, fontweight='bold')

# Customize the plot appearance
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig(f"{file_save_path}performance_by_model_boxplot.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Create comprehensive statistical summary
summary_table = df_benchmark.groupby('method')['model_performance'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('25%', lambda x: x.quantile(0.25)),
    ('median', 'median'),
    ('75%', lambda x: x.quantile(0.75)),
    ('max', 'max')
]).round(4)

print("Performance Statistics by Feature Selection Method:")
print(summary_table)

# %% [markdown]
# ### Performance vs. Feature Set Size (k value)

# %%
# Derive feature_set_sizes from the dataframe to handle different runs
feature_set_sizes_viz = sorted(df_benchmark['k_value'].unique())

# Calculate mean and standard deviation for each method and k value
k_performance_stats = df_benchmark.groupby(['method', 'k_value'])['model_performance'].agg(['mean', 'std', 'count']).reset_index()

# Create publication-quality line plot with standard deviation bands
plt.figure(figsize=(10, 6), dpi=300)
plt.rcParams['font.family'] = 'sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# Define publication-quality color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond

# Plot each method with error bands
for i, method in enumerate(k_performance_stats['method'].unique()):
    method_data = k_performance_stats[k_performance_stats['method'] == method]
    
    # Sort by k_value to ensure proper line plotting
    method_data = method_data.sort_values('k_value')
    
    # Plot the mean line
    plt.plot(method_data['k_value'], method_data['mean'], 
             marker=markers[i], linewidth=2.5, markersize=8, 
             color=colors[i], markeredgecolor='white', markeredgewidth=1,
             label=method_labels.get(method, method))
    
    # Add standard deviation bands (shaded area)
    plt.fill_between(method_data['k_value'], 
                     method_data['mean'] - method_data['std'],
                     method_data['mean'] + method_data['std'],
                     alpha=0.15, color=colors[i])

plt.title('Feature Selection Performance vs. Number of Features Selected', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Number of Features Selected (k)', fontsize=14, fontweight='bold')
plt.ylabel('Mean R² Score ± Std. Dev.', fontsize=14, fontweight='bold')
plt.xscale('log')  # Use log scale for better visualization of wide k range
plt.xticks(feature_set_sizes_viz, feature_set_sizes_viz, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.2, linestyle='--')
plt.legend(title='Feature Selection Method', fontsize=11, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{file_save_path}performance_vs_k_value.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Create publication-quality faceted line plots by model type
plt.figure(figsize=(18, 6), dpi=300)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# Calculate stats by method, k_value, and model_name
model_k_stats = df_benchmark.groupby(['method', 'k_value', 'model_name'])['model_performance'].agg(['mean', 'std']).reset_index()

# Create subplots for each model
models = ['KNeighborsRegressor', 'LinearRegression', 'SVR']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define publication-quality color palette and markers
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond

for i, model in enumerate(models):
    model_data = model_k_stats[model_k_stats['model_name'] == model]
    
    for j, method in enumerate(model_data['method'].unique()):
        method_model_data = model_data[model_data['method'] == method].sort_values('k_value')
        
        axes[i].plot(method_model_data['k_value'], method_model_data['mean'], 
                     marker=markers[j], linewidth=2.5, markersize=6,
                     color=colors[j], markeredgecolor='white', markeredgewidth=1,
                     label=method_labels.get(method, method))
        
        axes[i].fill_between(method_model_data['k_value'],
                            method_model_data['mean'] - method_model_data['std'],
                            method_model_data['mean'] + method_model_data['std'],
                            alpha=0.15, color=colors[j])
    
    axes[i].set_title(f'{model}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Number of Features Selected (k)', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Mean R² Score', fontsize=12, fontweight='bold')
    axes[i].set_xscale('log')
    axes[i].set_xticks(feature_set_sizes_viz, feature_set_sizes_viz, fontsize=10)
    axes[i].tick_params(axis='y', labelsize=10)
    axes[i].grid(True, alpha=0.2, linestyle='--')
    axes[i].legend(fontsize=10, framealpha=0.9)

plt.suptitle('Feature Selection Performance vs. k Value by ML Model', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{file_save_path}performance_vs_k_by_model.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Statistical analysis of k value effect
print("Performance Trend Analysis by k Value:")
for method in df_benchmark['method'].unique():
    method_data = df_benchmark[df_benchmark['method'] == method]
    
    # Calculate correlation between k_value and performance
    correlation = method_data['k_value'].corr(method_data['model_performance'])
    
    # Calculate performance change from smallest to largest k
    k_min_perf = method_data[method_data['k_value'] == 10]['model_performance'].mean()
    k_max_perf = method_data[method_data['k_value'] == 1280]['model_performance'].mean()
    performance_change = k_max_perf - k_min_perf
    
    print(f"\n{method_labels.get(method, method)}:")
    print(f"  Correlation (k vs performance): {correlation:.4f}")
    print(f"  Performance change (k=10 to k=1280): {performance_change:.4f}")
    print(f"  Optimal k range: {method_data.groupby('k_value')['model_performance'].mean().idxmax()} features")

# %% [markdown]
# ### Performance vs. Feature Set Size (k value) - Excluding Specific k Values

# %%
# Create visualization excluding specific k values (e.g., k=500)
k_values_to_exclude = [500]  # Add any k values you want to exclude here

# Filter dataframe to exclude specific k values
df_filtered = df_benchmark[~df_benchmark['k_value'].isin(k_values_to_exclude)]

# Derive feature_set_sizes from the filtered dataframe
feature_set_sizes_filtered = sorted(df_filtered['k_value'].unique())

# Calculate mean and standard deviation for each method and k value
k_performance_stats_filtered = df_filtered.groupby(['method', 'k_value'])['model_performance'].agg(['mean', 'std', 'count']).reset_index()

# Create publication-quality line plot with standard deviation bands (excluding specific k values)
plt.figure(figsize=(10, 6), dpi=300)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# Define publication-quality color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond

# Plot each method with error bands
for i, method in enumerate(k_performance_stats_filtered['method'].unique()):
    method_data = k_performance_stats_filtered[k_performance_stats_filtered['method'] == method]
    
    # Sort by k_value to ensure proper line plotting
    method_data = method_data.sort_values('k_value')
    
    # Plot the mean line
    plt.plot(method_data['k_value'], method_data['mean'], 
             marker=markers[i], linewidth=2.5, markersize=8, 
             color=colors[i], markeredgecolor='white', markeredgewidth=1,
             label=method_labels.get(method, method))
    
    # Add standard deviation bands (shaded area)
    plt.fill_between(method_data['k_value'], 
                     method_data['mean'] - method_data['std'],
                     method_data['mean'] + method_data['std'],
                     alpha=0.15, color=colors[i])

plt.title(f'Feature Selection Performance vs. Number of Features Selected\n(Excluding k={k_values_to_exclude})', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Number of Features Selected (k)', fontsize=14, fontweight='bold')
plt.ylabel('Mean R² Score ± Std. Dev.', fontsize=14, fontweight='bold')
plt.xscale('log')  # Use log scale for better visualization of wide k range
plt.xticks(feature_set_sizes_filtered, feature_set_sizes_filtered, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.2, linestyle='--')
plt.legend(title='Feature Selection Method', fontsize=11, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{file_save_path}performance_vs_k_value_excluding_{'_'.join(map(str, k_values_to_exclude))}.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Created visualization excluding k values: {k_values_to_exclude}")
print(f"Remaining k values in visualization: {feature_set_sizes_filtered}")

# %% [markdown]
# ### Performance vs. Feature Set Size (k value) - Without Standard Error Bars

# %%
# Create line plot without standard error bars (cleaner visualization)
plt.figure(figsize=(10, 6), dpi=300)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# Define publication-quality color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond

# Plot each method without error bands
for i, method in enumerate(k_performance_stats_filtered["method"].unique()):
    method_data = k_performance_stats_filtered[k_performance_stats_filtered["method"] == method]
    
    # Sort by k_value to ensure proper line plotting
    method_data = method_data.sort_values('k_value')
    
    # Plot the mean line only (no error bands)
    plt.plot(method_data['k_value'], method_data['mean'], 
             marker=markers[i], linewidth=2.5, markersize=8, 
             color=colors[i], markeredgecolor='white', markeredgewidth=1,
             label=method_labels.get(method, method))

plt.title('Feature Selection Performance vs. Number of Features Selected', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Number of Features Selected (k)', fontsize=14, fontweight='bold')
plt.ylabel('Mean R² Score', fontsize=14, fontweight='bold')
plt.xscale('log')  # Use log scale for better visualization of wide k range
plt.xticks(feature_set_sizes_viz, feature_set_sizes_viz, fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(left=9, right=170)
plt.grid(True, alpha=0.2, linestyle='--')
plt.legend(title='Feature Selection Method', fontsize=14, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{file_save_path}performance_vs_k_value_no_error_bars.png", dpi=300, bbox_inches='tight')
plt.show()

print("Created line plot without standard error bars")
