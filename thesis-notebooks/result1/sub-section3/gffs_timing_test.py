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
from PathLoader import PathLoader  # noqa: E402

path_loader = PathLoader("data_config.env", "current_user.env")

# %%
from DataLink import DataLink  # noqa: E402

data_link = DataLink(path_loader, "data_codes.csv")

# %%
folder_name = "ThesisResult4-FeatureSelectionBenchmark"
exp_id = "v1"

if not os.path.exists(f"{path_loader.get_data_path()}data/results/{folder_name}"):
    os.makedirs(f"{path_loader.get_data_path()}data/results/{folder_name}")

file_save_path = f"{path_loader.get_data_path()}data/results/{folder_name}/"


# %% [markdown]
# ## GFFS Timing Test

# %%
import time # noqa: F811, E402
import numpy as np # noqa: E402
import pandas as pd # noqa: E402
import matplotlib.pyplot as plt # noqa: E402
from scipy.optimize import curve_fit # noqa: E402
from sklearn.svm import SVR # noqa: E402
from sklearn.feature_selection import f_regression # noqa: E402
from toolkit import greedy_feedforward_select # noqa: E402


# %%
def get_most_correlated_feature(X: pd.DataFrame, y: pd.Series) -> str:
    """Find the feature with highest correlation to target"""
    correlations = []
    for col in X.columns:
        corr = np.corrcoef(X[col], y)[0, 1]
        correlations.append((col, abs(corr)))
    
    # Return feature with highest absolute correlation
    most_correlated = max(correlations, key=lambda x: x[1])
    return most_correlated[0]


# %%
def time_gffs_selection(X, y, k_values, iterations=3):
    """Time GFFS feature selection for different k values"""
    timing_results = []

    # Create SVR model with linear kernel
    svr_model = SVR(kernel='linear', C=1.0)
    
    for k in k_values:
        iteration_times = []

        for i in range(iterations):
            start_time = time.time()

            # Find starting feature (most correlated with target)
            start_feature = get_most_correlated_feature(X, y)
            
            # Pure GFFS selection without any other processing
            selected_features = greedy_feedforward_select(
                X, y, k, svr_model, start_feature, cv=5, scoring_method='r2', verbose=0
            )

            end_time = time.time()
            selection_time = end_time - start_time
            iteration_times.append(selection_time)

            print(f"k={k}, iteration {i + 1}: {selection_time:.4f}s")

        # Calculate statistics
        mean_time = np.mean(iteration_times)
        std_time = np.std(iteration_times)

        timing_results.append(
            {
                "k_value": k,
                "mean_time": mean_time,
                "std_time": std_time,
                "min_time": min(iteration_times),
                "max_time": max(iteration_times),
                "n_features_selected": len(selected_features),
            }
        )

        print(f"k={k}: Mean time = {mean_time:.4f}s ± {std_time:.4f}s")

    return pd.DataFrame(timing_results)


# %%
def analyze_gffs_complexity(timing_df):
    """Analyze time complexity of GFFS feature selection"""
    
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c

    def linear_func(x, a, b):
        return a * x + b

    # Fit different complexity models
    x_data = timing_df["k_value"]
    y_data = timing_df["mean_time"]

    try:
        # Quadratic fit (O(n²) complexity)
        popt_quad, _ = curve_fit(quadratic_func, x_data, y_data)
        y_pred_quad = quadratic_func(x_data, *popt_quad)

        # Linear fit (O(n) complexity)
        popt_lin, _ = curve_fit(linear_func, x_data, y_data)
        y_pred_lin = linear_func(x_data, *popt_lin)

        # Calculate R² for both fits
        ss_res_quad = np.sum((y_data - y_pred_quad) ** 2)
        ss_tot_quad = np.sum((y_data - np.mean(y_data)) ** 2)
        r2_quad = 1 - (ss_res_quad / ss_tot_quad)

        ss_res_lin = np.sum((y_data - y_pred_lin) ** 2)
        ss_tot_lin = np.sum((y_data - np.mean(y_data)) ** 2)
        r2_lin = 1 - (ss_res_lin / ss_tot_lin)

        print(f"Quadratic fit R²: {r2_quad:.4f}")
        print(f"Linear fit R²: {r2_lin:.4f}")

        # Plot fits
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, "bo-", label="Actual Data")
        plt.plot(x_data, y_pred_quad, "r--", label=f"Quadratic Fit (R²={r2_quad:.3f})")
        plt.plot(x_data, y_pred_lin, "g--", label=f"Linear Fit (R²={r2_lin:.3f})")
        plt.title("GFFS Time Complexity Analysis")
        plt.xlabel("Number of Features Selected (k)")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return {
            "quadratic_r2": r2_quad,
            "linear_r2": r2_lin,
            "quadratic_params": popt_quad,
            "linear_params": popt_lin
        }

    except Exception as e:
        print(f"Complexity analysis failed: {e}")
        return None


# %%
def plot_gffs_timing(timing_df):
    """Plot GFFS timing results"""
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        timing_df["k_value"],
        timing_df["mean_time"],
        yerr=timing_df["std_time"],
        fmt="o-",
        capsize=5,
    )
    plt.title("GFFS Feature Selection Time vs Number of Features")
    plt.xlabel("Number of Features Selected (k)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
def run_gffs_timing_test(feature_data, label_data, k_values=None, iterations=3, feature_subset_size=100):
    """Run complete GFFS timing test"""
    
    if k_values is None:
        k_values = list(range(1, 21))  # k=1 to 20
    
    # Prepare data subset for faster testing (GFFS is computationally intensive)
    feature_subset = feature_data.iloc[:, :feature_subset_size]
    print(f"Testing GFFS on subset: {feature_subset.shape}")
    
    # Run timing test
    print("Starting isolated GFFS timing test...")
    gffs_timing_df = time_gffs_selection(
        feature_subset, label_data, k_values, iterations
    )
    
    # Display results
    print("\nGFFS Timing Results:")
    print(gffs_timing_df.round(4))
    
    # Plot results
    plot_gffs_timing(gffs_timing_df)
    
    # Analyze complexity
    complexity_results = analyze_gffs_complexity(gffs_timing_df)
    
    return gffs_timing_df, complexity_results


# %%
def test_gffs_scalability(feature_data, label_data, k_values, feature_subset_size=100):
    """
    Test GFFS scalability with different k values
    """
    # Use small subset for scalability testing (GFFS is computationally intensive)
    feature_subset = feature_data.iloc[:, :feature_subset_size]
    print(f"Testing GFFS scalability on subset: {feature_subset.shape}")
    print(f"Testing k values: {k_values}")
    
    scalability_results = []
    
    # Create SVR model with linear kernel
    svr_model = SVR(kernel='linear', C=1.0)
    
    for k in k_values:
        print(f"\n--- Testing k={k} ---")
        
        # Find starting feature
        start_feature = get_most_correlated_feature(feature_subset, label_data)
        
        # Test GFFS
        start_time = time.time()
        selected_features = greedy_feedforward_select(
            feature_subset, label_data, k, svr_model, start_feature, cv=5, scoring_method='r2', verbose=0
        )
        gffs_time = time.time() - start_time
        
        print(f"GFFS time: {gffs_time:.4f}s")
        print(f"GFFS selected (first 10): {selected_features[:10]}")
        
        scalability_results.append({
            "k": k,
            "gffs_time": gffs_time,
            "gffs_selected": selected_features,
            "start_feature": start_feature
        })
    
    return pd.DataFrame(scalability_results)


# %% [markdown]
# ## Execution

# %% [markdown]
# ### Loading data

# %%
# Load Proteomics Palbociclib dataset for GFFS timing test
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
# ### Test 1: Basic GFFS Timing Test

# %%
# Setup test parameters
k_values_to_test = list(range(1, 21))  # k=1 to 20
n_iterations = 3  # Fewer iterations due to GFFS computational intensity
feature_subset_size = 100  # Use first 100 features for faster testing

# %%
# Run the GFFS timing test
print("Executing GFFS timing test...")
gffs_timing_df, complexity_results = run_gffs_timing_test(
    feature_data, 
    label_data, 
    k_values=k_values_to_test, 
    iterations=n_iterations, 
    feature_subset_size=feature_subset_size
)

# %%
# Save GFFS timing results
gffs_timing_df.to_csv(f"{file_save_path}gffs_timing_test_{exp_id}.csv")
print(f"GFFS timing results saved to: {file_save_path}gffs_timing_test_{exp_id}.csv")

# %%
# Display final summary
print("\nGFFS Timing Test Summary:")
print(f"Tested k values: {k_values_to_test}")
print(f"Number of iterations per k: {n_iterations}")
print(f"Feature subset size: {feature_subset_size}")
print(f"Total timing runs: {len(k_values_to_test) * n_iterations}")
print(f"Total execution time: {gffs_timing_df['mean_time'].sum():.2f}s")

# %% [markdown]
# ### Test 2: GFFS Scalability Test

# %%
# GFFS Scalability Test
print("\n" + "="*60)
print("GFFS SCALABILITY TEST")
print("="*60)

# Test with different k values and small feature set
k_values_scalability = [5, 10, 15, 20]  # Test with these k values
feature_subset_scalability = 100  # Use first 100 features

print(f"Running GFFS scalability test with {feature_subset_scalability} features")
print(f"Testing k values: {k_values_scalability}")

# Run the scalability test
scalability_df = test_gffs_scalability(
    feature_data, 
    label_data, 
    k_values=k_values_scalability,
    feature_subset_size=feature_subset_scalability
)

# Save scalability results
scalability_df.to_csv(f"{file_save_path}gffs_scalability_test_{exp_id}.csv")
print(f"\nGFFS scalability results saved to: {file_save_path}gffs_scalability_test_{exp_id}.csv")

# Display scalability summary
print("\nScalability Results Summary:")
print(f"GFFS tested with {feature_subset_scalability} features")
print(f"Tested k values: {k_values_scalability}")

# Show timing progression
print("\nTiming Progression:")
for _, row in scalability_df.iterrows():
    print(f"k={row['k']}: GFFS={row['gffs_time']:.4f}s")

# Analyze time complexity
print("\nTime Complexity Analysis:")
k_values = scalability_df['k'].values
gffs_times = scalability_df['gffs_time'].values

# Simple linear regression to understand scaling
if len(k_values) > 1:
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(k_values, gffs_times)
    print(f"Time vs k linear fit: time = {slope:.4f} * k + {intercept:.4f}")
    print(f"R² = {r_value**2:.4f}")
    print(f"Approximate time per feature: {slope:.4f}s")

# %%
def plot_gffs_scalability_results(scalability_df, feature_subset_size=100):
    """
    Plot scalability results for GFFS
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Time vs k
    k_values = scalability_df['k'].values
    gffs_times = scalability_df['gffs_time'].values
    
    ax1.plot(k_values, gffs_times, 'bo-', linewidth=2, markersize=8)
    ax1.set_title(f'GFFS Execution Time vs k\n({feature_subset_size} features)')
    ax1.set_xlabel('Number of Features Selected (k)')
    ax1.set_ylabel('Time (seconds)')
    ax1.grid(True, alpha=0.3)
    
    # Add linear regression line
    if len(k_values) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(k_values, gffs_times)
        x_fit = np.linspace(min(k_values), max(k_values), 100)
        y_fit = slope * x_fit + intercept
        ax1.plot(x_fit, y_fit, 'r--', label=f'Linear fit (R²={r_value**2:.3f})')
        ax1.legend()
    
    # Plot 2: Time per feature vs k
    time_per_feature = gffs_times / k_values
    
    ax2.plot(k_values, time_per_feature, 'go-', linewidth=2, markersize=8)
    ax2.set_title('GFFS Time per Feature vs k')
    ax2.set_xlabel('Number of Features Selected (k)')
    ax2.set_ylabel('Time per Feature (seconds)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plot_filename = f"{file_save_path}gffs_scalability_plot_{exp_id}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"GFFS scalability plot saved to: {plot_filename}")

# %%
# Plot scalability results
print("\n" + "="*60)
print("PLOTTING GFFS SCALABILITY RESULTS")
print("="*60)

plot_gffs_scalability_results(scalability_df, feature_subset_size=feature_subset_scalability)

# Additional analysis
print("\nAdditional Scalability Analysis:")
print(f"Fastest execution: k={scalability_df.loc[scalability_df['gffs_time'].idxmin(), 'k']} "
      f"({scalability_df['gffs_time'].min():.4f}s)")
print(f"Slowest execution: k={scalability_df.loc[scalability_df['gffs_time'].idxmax(), 'k']} "
      f"({scalability_df['gffs_time'].max():.4f}s)")
print(f"Total time for all k values: {scalability_df['gffs_time'].sum():.4f}s")

# Calculate efficiency metrics
if len(k_values) > 1:
    efficiency_ratio = scalability_df['gffs_time'].iloc[-1] / scalability_df['gffs_time'].iloc[0]
    print(f"Efficiency ratio (k=20 vs k=5): {efficiency_ratio:.2f}x")

# %% [markdown]
# ### Test 3: GFFS Performance Comparison

# %%
def compare_gffs_with_other_methods(feature_data, label_data, k_values, feature_subset_size=50):
    """
    Compare GFFS with other feature selection methods on small dataset
    """
    from toolkit import f_regression_select, mrmr_select_fcq_fast, mutual_information_select
    
    # Use very small subset for comparison (GFFS is slow)
    feature_subset = feature_data.iloc[:, :feature_subset_size]
    print(f"Comparing methods on subset: {feature_subset.shape}")
    
    comparison_results = []
    
    # Create SVR model for GFFS
    svr_model = SVR(kernel='linear', C=1.0)
    
    for k in k_values:
        print(f"\n--- Testing k={k} ---")
        
        # Test ANOVA (F-regression)
        start_time = time.time()
        anova_selected, anova_scores = f_regression_select(feature_subset, label_data, k)
        anova_time = time.time() - start_time
        
        # Test MRMR
        start_time = time.time()
        mrmr_selected, mrmr_scores = mrmr_select_fcq_fast(feature_subset, label_data, k)
        mrmr_time = time.time() - start_time
        
        # Test Mutual Information
        start_time = time.time()
        mi_selected, mi_scores = mutual_information_select(feature_subset, label_data, k)
        mi_time = time.time() - start_time
        
        # Test GFFS
        start_feature = get_most_correlated_feature(feature_subset, label_data)
        start_time = time.time()
        gffs_selected = greedy_feedforward_select(
            feature_subset, label_data, k, svr_model, start_feature, cv=5, scoring_method='r2', verbose=0
        )
        gffs_time = time.time() - start_time
        
        print(f"ANOVA time: {anova_time:.4f}s")
        print(f"MRMR time: {mrmr_time:.4f}s")
        print(f"MI time: {mi_time:.4f}s")
        print(f"GFFS time: {gffs_time:.4f}s")
        print(f"GFFS is {gffs_time/anova_time:.1f}x slower than ANOVA")
        
        comparison_results.append({
            "k": k,
            "anova_time": anova_time,
            "mrmr_time": mrmr_time,
            "mi_time": mi_time,
            "gffs_time": gffs_time,
            "gffs_vs_anova_ratio": gffs_time / anova_time
        })
    
    return pd.DataFrame(comparison_results)

# %%
# Method Comparison Test
print("\n" + "="*60)
print("METHOD COMPARISON TEST")
print("="*60)

# Test with very small k values and feature set
k_values_comparison = [5, 10]  # Small k values only
feature_subset_comparison = 50  # Very small feature set

print(f"Running method comparison with {feature_subset_comparison} features")
print(f"Testing k values: {k_values_comparison}")

# Run the comparison
comparison_df = compare_gffs_with_other_methods(
    feature_data, 
    label_data, 
    k_values=k_values_comparison,
    feature_subset_size=feature_subset_comparison
)

# Save comparison results
comparison_df.to_csv(f"{file_save_path}gffs_comparison_test_{exp_id}.csv")
print(f"\nMethod comparison results saved to: {file_save_path}gffs_comparison_test_{exp_id}.csv")

# Display comparison summary
print("\nComparison Results Summary:")
print(f"Average GFFS vs ANOVA time ratio: {comparison_df['gffs_vs_anova_ratio'].mean():.1f}x")
print(f"GFFS is significantly slower than other methods as expected")

# Show detailed results
print("\nDetailed Results:")
for _, row in comparison_df.iterrows():
    print(f"k={row['k']}: "
          f"ANOVA={row['anova_time']:.4f}s, "
          f"MRMR={row['mrmr_time']:.4f}s, "
          f"MI={row['mi_time']:.4f}s, "
          f"GFFS={row['gffs_time']:.4f}s, "
          f"GFFS/ANOVA={row['gffs_vs_anova_ratio']:.1f}x")
