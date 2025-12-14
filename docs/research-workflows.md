# Research Workflow Examples

## Overview

This document provides practical examples of common research workflows using the dynmarker framework. Each workflow includes:
- Specific research questions
- Step-by-step implementation
- Expected outputs and analysis
- Troubleshooting tips

## Workflow 1: Basic Drug Response Biomarker Discovery

### Research Question
"Which proteomic features best predict Palbociclib drug response in cancer cell lines?"

### Implementation Steps

#### 1. Data Preparation
```python
from PathLoader import PathLoader
from DataFunctions import create_joint_dataset_from_proteome_gdsc, create_feature_and_label
import pickle

# Load data
path_loader = PathLoader('data_config.env', 'current_user.env')
data_path = path_loader.get_data_path()

with open(f'{data_path}data/drug-response/GDSC2/cache_gdsc2.pkl', 'rb') as f:
    gdsc2 = pickle.load(f)
    gdsc2_info = pickle.load(f)

with open(f'{data_path}data/proteomic-expression/goncalves-2022-cell/goncalve_proteome_fillna_processed.pkl', 'rb') as f:
    proteome_data = pickle.load(f)

# Create joint dataset
target_drug = "Palbociclib"
target_variable = "LN_IC50"
joint_data = create_joint_dataset_from_proteome_gdsc(target_drug, proteome_data, gdsc2, drug_value=target_variable)

# Split features and labels
X, y = create_feature_and_label(joint_data, label_name=target_variable)
```

#### 2. Quick Baseline Evaluation
```python
from dynmarker.FeatureSelection import naive_test_regression

# Test basic ML models
baseline_results = naive_test_regression(X, y, cv=5, verbose=1)
print("Baseline performance established")
```

#### 3. Feature Selection with MRMR
```python
from dynmarker.FeatureSelection import mrmr_select_fcq

# Select top 15 features
selected_features, mrmr_scores = mrmr_select_fcq(X, y, K=15, verbose=1)

# Extract selected feature data
X_selected = X.iloc[:, selected_features]
```

#### 4. Model Evaluation with Selected Features
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Evaluate with selected features
model = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(model, X_selected, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores

print(f"Mean MSE with MRMR features: {np.mean(mse_scores):.4f} (+/- {np.std(mse_scores):.4f})")
```

#### 5. Feature Importance Analysis
```python
import pandas as pd

# Train final model and get feature importances
model.fit(X_selected, y)
feature_importance = pd.DataFrame({
    'feature': X_selected.columns,
    'importance': model.feature_importances_,
    'mrmr_score': mrmr_scores
}).sort_values('importance', ascending=False)

print("Top predictive features:")
print(feature_importance.head(10))
```

### Expected Outputs
- Baseline model performance metrics
- MRMR-selected feature list with scores
- Final model performance with selected features
- Feature importance rankings

## Workflow 2: Comparative Feature Selection Benchmarking

### Research Question
"How do different feature selection methods compare for proteomic biomarker discovery?"

### Implementation Steps

#### 1. Setup Benchmarking Environment
```python
from benchmark_feature_selection_batch import run_feature_selection_benchmark
import pandas as pd
import numpy as np

# Define comparison parameters
feature_sizes = [5, 10, 15, 20, 25]
algorithms = ['mrmr', 'relieff', 'enet', 'f_regression']
models = ['linear', 'svr', 'rf']
```

#### 2. Run Comprehensive Benchmark
```python
# Run benchmark (commented for safety - uncomment with actual data)
# benchmark_results = run_feature_selection_benchmark(
#     X, y,
#     feature_sizes=feature_sizes,
#     algorithms=algorithms,
#     models=models,
#     cv_folds=5,
#     n_iterations=10,
#     n_cores=4
# )
```

#### 3. Analyze Results
```python
def analyze_benchmark_results(results):
    # Performance comparison by algorithm
    algorithm_performance = results.groupby(['algorithm', 'feature_size'])['performance'].agg(['mean', 'std'])
    
    # Stability analysis
    feature_stability = analyze_feature_selection_stability(results)
    
    return algorithm_performance, feature_stability

# algorithm_perf, stability = analyze_benchmark_results(benchmark_results)
```

#### 4. Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_benchmark_comparison(algorithm_performance):
    plt.figure(figsize=(12, 8))
    for algorithm in algorithm_performance.index.levels[0]:
        algorithm_data = algorithm_performance.loc[algorithm]
        plt.errorbar(algorithm_data.index, algorithm_data['mean'], 
                    yerr=algorithm_data['std'], label=algorithm, marker='o')
    
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Performance (Negative MSE)')
    plt.title('Feature Selection Algorithm Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# plot_benchmark_comparison(algorithm_performance)
```

### Expected Outputs
- Performance metrics for each algorithm-model combination
- Feature selection stability measures
- Comparative visualization plots
- Statistical significance testing results

## Workflow 3: Dynamic Feature Integration

### Research Question
"Do dynamic features from ODE simulations improve biomarker prediction compared to static expression data?"

### Implementation Steps

#### 1. Generate Dynamic Features
```python
from UTIL_create_dynamic_features import create_dynamic_features
from UTIL_create_initial_conditions import generate_initial_conditions

# Generate initial conditions for ODE system
initial_conditions = generate_initial_conditions(
    proteome_data,  # Use proteomic data as baseline
    system_parameters={'time_points': 100, 'step_size': 0.1}
)

# Simulate dynamic features
dynamic_features = create_dynamic_features(
    initial_conditions,
    ode_parameters={'simulation_time': 10, 'output_points': 50},
    feature_extraction_method='summary_statistics'
)
```

#### 2. Combine Static and Dynamic Features
```python
# Ensure proper alignment (same cell lines)
static_features = X  # From previous workflow

# Combine features
combined_features = pd.concat([static_features, dynamic_features], axis=1)

# Handle potential missing values
combined_features = combined_features.fillna(combined_features.mean())
```

#### 3. Comparative Evaluation
```python
from dynmarker import EvaluationPipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def compare_feature_types(feature_sets, feature_names, y, model_list, cv_folds=5):
    results = {}
    
    for name, features in zip(feature_names, feature_sets):
        pipeline = EvaluationPipeline()
        pipeline.set_function(example_run_model_func)
        
        pipeline.run_function(
            features, y, 
            iterations=5, 
            k_ranges=[10, 15, 20], 
            model_list=model_list,
            n_fold_splits=cv_folds,
            n_cores=2
        )
        
        results[name] = pipeline.evaluation_df
    
    return results

# feature_sets = [static_features, dynamic_features, combined_features]
# feature_names = ['static', 'dynamic', 'combined']
# model_list = [LinearRegression(), RandomForestRegressor()]

# comparison_results = compare_feature_types(feature_sets, feature_names, y, model_list)
```

#### 4. Dynamic Feature Analysis
```python
def analyze_dynamic_feature_contribution(comparison_results):
    # Extract performance metrics
    performance_summary = {}
    
    for feature_type, results_df in comparison_results.items():
        avg_performance = results_df['eval_score'].mean()
        performance_summary[feature_type] = avg_performance
    
    return performance_summary

# performance_summary = analyze_dynamic_feature_contribution(comparison_results)
```

### Expected Outputs
- Dynamic feature time series data
- Performance comparison: static vs dynamic vs combined features
- Contribution analysis of dynamic features
- Biological interpretation of dynamic patterns

## Workflow 4: Multi-drug Analysis Pipeline

### Research Question
"Are there common biomarkers that predict response to multiple CDK4/6 inhibitors?"

### Implementation Steps

#### 1. Multi-drug Data Preparation
```python
cdk46_inhibitors = ["Palbociclib", "Ribociclib", "Abemaciclib"]
multi_drug_data = {}

for drug in cdk46_inhibitors:
    joint_data = create_joint_dataset_from_proteome_gdsc(drug, proteome_data, gdsc2, drug_value="LN_IC50")
    X_drug, y_drug = create_feature_and_label(joint_data, 'LN_IC50')
    multi_drug_data[drug] = {'X': X_drug, 'y': y_drug}
```

#### 2. Consensus Feature Selection
```python
from dynmarker.FeatureSelection import mrmr_select_fcq

consensus_features = {}
feature_overlap = {}

for drug, data in multi_drug_data.items():
    features, scores = mrmr_select_fcq(data['X'], data['y'], K=15, verbose=0)
    consensus_features[drug] = features
    
    # Store feature names for overlap analysis
    feature_overlap[drug] = data['X'].columns[features].tolist()

# Find common features across drugs
common_features = set(feature_overlap[cdk46_inhibitors[0]])
for drug in cdk46_inhibitors[1:]:
    common_features = common_features.intersection(set(feature_overlap[drug]))

print(f"Common features across {len(cdk46_inhibitors)} CDK4/6 inhibitors: {len(common_features)}")
```

#### 3. Cross-drug Validation
```python
def cross_drug_validation(multi_drug_data, common_features):
    results = {}
    
    for train_drug in cdk46_inhibitors:
        for test_drug in cdk46_inhibitors:
            if train_drug != test_drug:
                # Train on one drug, test on another
                X_train = multi_drug_data[train_drug]['X']
                y_train = multi_drug_data[train_drug]['y']
                X_test = multi_drug_data[test_drug]['X']
                y_test = multi_drug_data[test_drug]['y']
                
                # Use common features
                X_train_common = X_train[list(common_features)]
                X_test_common = X_test[list(common_features)]
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_common, y_train)
                y_pred = model.predict(X_test_common)
                
                mse = mean_squared_error(y_test, y_pred)
                results[(train_drug, test_drug)] = mse
    
    return results

# cross_drug_results = cross_drug_validation(multi_drug_data, common_features)
```

### Expected Outputs
- Drug-specific feature selections
- Common biomarker identification
- Cross-drug prediction performance
- Biomarker stability metrics

## Workflow 5: Network-based Feature Selection

### Research Question
"Do protein interaction networks improve biomarker discovery by incorporating biological context?"

### Implementation Steps

#### 1. Network Data Integration
```python
with open(f'{data_path}data/protein-interaction/STRING/string_df.pkl', 'rb') as f:
    string_df = pickle.load(f)
    string_df_info = pickle.load(f)

# Map proteomic features to STRING network
protein_to_node_mapping = create_protein_network_mapping(proteome_data.columns, string_df)
```

#### 2. Network-constrained Feature Selection
```python
from benchmark_network import run_network_benchmark

# Run network-based feature selection
# network_results = run_network_benchmark(
#     X, y,
#     protein_network=string_df,
#     mapping=protein_to_node_mapping,
#     feature_sizes=[10, 15, 20],
#     network_methods=['neighborhood', 'pathway', 'centrality'],
#     n_iterations=5,
#     n_cores=4
# )
```

#### 3. Network Analysis
```python
def analyze_network_features(network_results):
    # Extract network-based feature importance
    network_importance = {}
    
    for method in network_results['network_methods']:
        method_results = network_results[method]
        # Analyze feature centrality, connectivity, etc.
        network_importance[method] = calculate_network_metrics(method_results)
    
    return network_importance

# network_importance = analyze_network_features(network_results)
```

### Expected Outputs
- Network-constrained feature selections
- Biological pathway enrichment analysis
- Network topology metrics for selected features
- Comparison with non-network methods

## Troubleshooting Common Issues

### Data Alignment Problems
**Issue**: Cell line mismatches between datasets
**Solution**: 
```python
# Verify alignment before joining
def verify_dataset_alignment(dataset1, dataset2, key_column='CELLLINE'):
    common_samples = set(dataset1[key_column]).intersection(set(dataset2[key_column]))
    print(f"Common samples: {len(common_samples)}")
    return common_samples
```

### Memory Optimization
**Issue**: Large datasets causing memory errors
**Solution**: Use memory-optimized scripts
```python
# Use memory-optimized versions
from benchmark_network_memory_optimized import run_network_benchmark_memopt
```

### Performance Tuning
**Issue**: Slow execution with large feature sets
**Solution**: Adjust parallel processing parameters
```python
# Start with fewer cores for debugging
pipeline.run_function(models, iterations=5, n_cores=2, verbose=1)

# Scale up once working
pipeline.run_function(models, iterations=20, n_cores=-1, verbose=0)
```

## Best Practices

### 1. Reproducibility
- Set random seeds for stochastic algorithms
- Document all parameters and versions
- Use version control for scripts and configurations

### 2. Validation
- Always use cross-validation
- Compare against appropriate baselines
- Perform statistical significance testing

### 3. Interpretation
- Consider biological context for feature selection
- Validate findings with external datasets
- Collaborate with domain experts for interpretation

### 4. Documentation
- Document each step of the workflow
- Save intermediate results for debugging
- Create comprehensive analysis reports

---

*These workflows provide starting points for common research scenarios. Adapt and extend them based on your specific research questions and data characteristics.*
