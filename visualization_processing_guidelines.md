# Visualization and Data Processing Guidelines for Feature Selection Benchmarking

## Overview

This document provides comprehensive guidelines for creating visualizations and processing results datasets in the feature selection benchmarking project. These guidelines are based on lessons learned from analyzing `performance_visualisation.py`, `stability_analysis.py`, and `compute_analysis.py`.

## 1. Dataset Specifications

### Core Data Structure
- Feature selection benchmark datasets are stored as pandas DataFrames in pickle format
- Key columns: `method`, `model_name`, `k_value`, `condition`, `model_performance`, `feature_selection_time`, `selected_features`
- The `condition` column follows format: `{method}_k{value}_{model}` (e.g., "mrmr_anova_prefilter_k5_KNeighborsRegressor")

### Expected Data Types
- **Methods**: string identifiers (e.g., 'anova', 'mrmr', 'mutual', 'random')
- **Models**: string identifiers (e.g., 'KNeighborsRegressor', 'LinearRegression', 'SVR')
- **k_value**: integer representing number of features selected
- **Performance metrics**: float values (R², Pearson r, etc.)
- **Feature sets**: lists or arrays of feature names

### Common Analysis Patterns
- Multiple runs per method/model/k-value combination
- Performance comparison across methods and models
- Computational cost analysis (time complexity)
- Feature selection stability (Jaccard similarity)

## 2. Data Processing Principles

### Principle 1: Dynamic Detection Over Hardcoding
**Never hardcode specific identifiers. Always detect them from the dataset.**

```python
# ✅ CORRECT: Dynamic detection
feature_set_sizes = sorted(df_benchmark['k_value'].unique())
models = df_benchmark['model_name'].unique().tolist()
available_methods = df_benchmark['method'].unique().tolist()

# ❌ INCORRECT: Hardcoded assumptions
# feature_set_sizes = [5, 10, 20, 40]  # Don't do this!
# methods = ['gffs', 'mrmr']  # Don't do this!
```

### Principle 2: Always Parse Condition Strings
**The `condition` column is the source of truth for method, model, and k-value information.**

```python
def parse_condition_column(df_benchmark):
    """Parse condition column to extract method, k_value, and model_name"""
    parsed_data = []
    
    for idx, row in df_benchmark.iterrows():
        condition = row['condition']
        parts = condition.split('_')
        
        method_parts = []
        k_value = None
        model_name = None
        
        for part in parts:
            if part.startswith('k'):
                k_value = int(part[1:])  # Extract numeric value after 'k'
                method = '_'.join(method_parts)
                model_parts = parts[parts.index(part) + 1:]
                model_name = '_'.join(model_parts)
                break
            else:
                method_parts.append(part)
        
        # Handle edge cases gracefully
        if k_value is None:
            method = '_'.join(method_parts[:-1]) if len(method_parts) > 1 else method_parts[0]
            model_name = parts[-1] if parts else 'unknown'
            k_value = 0
        
        parsed_data.append({
            'condition': condition,
            'parsed_method': method,
            'parsed_k_value': k_value,
            'parsed_model_name': model_name
        })
    
    return pd.DataFrame(parsed_data)
```

### Principle 3: Robust Error Handling
**Always validate data and handle edge cases gracefully.**

```python
# Check for sufficient data before analysis
if len(df_benchmark['method'].unique()) < 2:
    print("Insufficient methods for comparison")
    return

# Handle missing or invalid data
for method in df_benchmark['method'].unique():
    method_data = df_benchmark[df_benchmark['method'] == method]
    if len(method_data) == 0:
        continue  # Skip empty method data
```

## 3. Visualization Design Principles

### Principle 4: Centralized Color and Marker Mapping
**Use consistent color and marker assignments across all visualizations.**

```python
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
```

### Principle 5: Dynamic Subplot Layout
**Calculate optimal subplot arrangement based on number of methods.**

```python
def get_dynamic_subplot_layout(n_items, max_cols=3):
    """Calculate optimal subplot layout based on number of items"""
    if n_items <= max_cols:
        return 1, n_items
    else:
        rows = (n_items + max_cols - 1) // max_cols  # Ceiling division
        return rows, max_cols
```

### Principle 6: Publication-Quality Styling
**Maintain consistent, professional visualization standards.**

```python
# Standard styling for all plots
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# High-resolution saving
plt.savefig(filename, dpi=300, bbox_inches='tight')
```

## 4. Code Architecture Patterns

### Pattern 1: Abstract Function Design
**Create functions that work with any dataset configuration.**

```python
def analyze_performance(df_benchmark, metric='model_performance'):
    """Analyze performance metrics for any dataset configuration"""
    methods = df_benchmark['method'].unique()
    results = {}
    
    for method in methods:
        method_data = df_benchmark[df_benchmark['method'] == method]
        results[method] = {
            'mean_performance': method_data[metric].mean(),
            'std_performance': method_data[metric].std(),
            'n_runs': len(method_data)
        }
    
    return results
```

### Pattern 2: Modular Visualization Components
**Break down complex visualizations into reusable components.**

```python
def create_performance_plot(df_benchmark, ax, metric='model_performance'):
    """Create performance comparison plot for any dataset"""
    color_mapping = get_consistent_color_mapping(df_benchmark['method'].unique())
    
    for method in df_benchmark['method'].unique():
        method_data = df_benchmark[df_benchmark['method'] == method]
        # Plot implementation here...
```

### Pattern 3: Statistical Comparison Framework
**Implement pairwise statistical comparisons for all methods.**

```python
def pairwise_statistical_comparison(df_benchmark, metric_column):
    """Perform pairwise statistical comparisons for all methods"""
    methods = df_benchmark['method'].unique()
    comparisons = []
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                method1_data = df_benchmark[df_benchmark['method'] == method1][metric_column]
                method2_data = df_benchmark[df_benchmark['method'] == method2][metric_column]
                
                if len(method1_data) > 0 and len(method2_data) > 0:
                    t_stat, p_value = ttest_ind(method1_data, method2_data, equal_var=False)
                    comparisons.append({
                        'method1': method1,
                        'method2': method2,
                        't_statistic': t_stat,
                        'p_value': p_value
                    })
    
    return comparisons
```

## 5. Best Practices for Dataset Processing

### Practice 1: Data Validation
**Always validate dataset structure before processing.**

```python
def validate_benchmark_dataset(df_benchmark):
    """Validate feature selection benchmark dataset structure"""
    required_columns = ['method', 'model_name', 'k_value', 'condition', 'model_performance']
    
    for col in required_columns:
        if col not in df_benchmark.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check for sufficient data
    if len(df_benchmark) == 0:
        raise ValueError("Dataset is empty")
    
    return True
```

### Practice 2: Method Label Generation
**Create appropriate labels for methods dynamically.**

```python
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
```

### Practice 3: Comprehensive Reporting
**Generate detailed reports with proper formatting.**

```python
def save_and_print(message, report_file=None, level="info"):
    """Print message to console and save to report file with proper formatting"""
    print(message)
    
    if report_file:
        if level == "header":
            report_file.write(f"# {message}\n\n")
        elif level == "section":
            report_file.write(f"## {message}\n\n")
        elif level == "subsection":
            report_file.write(f"### {message}\n\n")
        else:
            report_file.write(f"{message}\n\n")
```

## 6. Implementation Examples

### Example 1: Complete Analysis Pipeline

```python
def run_complete_analysis(df_benchmark, file_save_path, exp_id):
    """Complete analysis pipeline following all guidelines"""
    
    # 1. Validate dataset
    validate_benchmark_dataset(df_benchmark)
    
    # 2. Parse condition column
    df_benchmark = parse_condition_column(df_benchmark)
    
    # 3. Dynamic detection
    methods = df_benchmark['method'].unique()
    models = df_benchmark['model_name'].unique()
    k_values = sorted(df_benchmark['k_value'].unique())
    
    # 4. Generate consistent mappings
    color_mapping = get_consistent_color_mapping(methods)
    marker_mapping = get_consistent_marker_mapping(methods)
    method_labels = generate_method_labels(methods)
    
    # 5. Perform analyses
    performance_results = analyze_performance(df_benchmark)
    stability_results = analyze_stability(df_benchmark)
    statistical_comparisons = pairwise_statistical_comparison(df_benchmark, 'model_performance')
    
    # 6. Create visualizations
    create_comprehensive_visualizations(df_benchmark, file_save_path, exp_id)
    
    # 7. Generate report
    generate_analysis_report(performance_results, stability_results, statistical_comparisons)
```

### Example 2: Error-Resistant Visualization

```python
def create_error_resistant_plot(df_benchmark, plot_type):
    """Create visualization that handles edge cases gracefully"""
    methods = df_benchmark['method'].unique()
    
    if len(methods) == 0:
        print("No methods found in dataset")
        return
    
    color_mapping = get_consistent_color_mapping(methods)
    
    for method in methods:
        method_data = df_benchmark[df_benchmark['method'] == method]
        
        if len(method_data) == 0:
            print(f"No data for method: {method}")
            continue
        
        # Safe plotting logic here...
```

## 7. Common Pitfalls to Avoid

### Pitfall 1: Hardcoded Method Assumptions
**❌ Don't assume specific methods exist**
```python
# ❌ INCORRECT
if 'gffs' in df_benchmark['method'].unique():
    gffs_data = df_benchmark[df_benchmark['method'] == 'gffs']

# ✅ CORRECT
methods = df_benchmark['method'].unique()
for method in methods:
    method_data = df_benchmark[df_benchmark['method'] == method]
```

### Pitfall 2: Fixed Color Assignments
**❌ Don't use fixed color arrays**
```python
# ❌ INCORRECT
colors = ['blue', 'orange', 'green', 'red']  # Will break with >4 methods

# ✅ CORRECT
color_mapping = get_consistent_color_mapping(methods)
```

### Pitfall 3: Static Layouts
**❌ Don't assume fixed number of subplots**
```python
# ❌ INCORRECT
fig, axes = plt.subplots(2, 2)  # Assumes exactly 4 methods

# ✅ CORRECT
rows, cols = get_dynamic_subplot_layout(len(methods))
fig, axes = plt.subplots(rows, cols)
```

## 8. Summary

These guidelines ensure that all feature selection benchmarking analyses are:
- **Dataset-agnostic**: Work with any configuration of methods, models, and parameters
- **Scalable**: Handle varying numbers of methods and data points
- **Consistent**: Maintain uniform styling and analysis patterns
- **Robust**: Gracefully handle edge cases and missing data
- **Maintainable**: Use modular, reusable code patterns

By following these principles, future analyses will be more reliable, adaptable, and professional.

---

**Last Updated**: 2025-10-16  
**Based on Analysis of**: `performance_visualisation.py`, `stability_analysis.py`, `compute_analysis.py`
