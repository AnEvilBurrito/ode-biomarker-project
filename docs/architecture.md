# Architecture Overview

## System Architecture

The dynmarker project follows a modular architecture designed for flexible biomarker discovery research. The system integrates data loading, feature engineering, machine learning, and evaluation components.

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  • Pipeline Scripts (SYPipelineScript.py, etc.)            │
│  • Notebooks (research analysis and visualization)         │
│  • Benchmarking Scripts                                     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Core Framework Layer                     │
├─────────────────────────────────────────────────────────────┤
│  • GeneralPipeline (dynmarker/GeneralPipeline.py)          │
│  • EvaluationPipeline (dynmarker/EvaluationPipeline.py)    │
│  • FeatureSelection (dynmarker/FeatureSelection.py)        │
│  • DataLoader (dynmarker/DataLoader.py)                    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Utility Layer                            │
├─────────────────────────────────────────────────────────────┤
│  • PathLoader (PathLoader.py)                              │
│  • DataFunctions (DataFunctions.py)                        │
│  • Toolkit (toolkit.py)                                    │
│  • Visualization (Visualisation.py)                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
├─────────────────────────────────────────────────────────────┤
│  • External Data Repository                                │
│  • Configuration Files (data_config.env, current_user.env) │
│  • Pickle Files (cached data)                              │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Management System

#### PathLoader
- **Purpose**: Flexible data path configuration management
- **Key Features**: Multi-user support, environment-based configuration
- **Usage**: Centralized data path resolution across the project

```python
from PathLoader import PathLoader
path_loader = PathLoader('data_config.env', 'current_user.env')
data_path = path_loader.get_data_path()
```

#### Data Integration
- **Supported Sources**: GDSC2 (drug response), CCLE (gene expression), proteomic data, STRING (protein interactions)
- **Integration Pattern**: Join datasets by cell line identifiers
- **Caching**: Pickle-based caching for performance

### 2. Pipeline Framework

#### GeneralPipeline Class
- **Role**: Flexible execution framework for custom functions
- **Features**: Parallel processing, result aggregation, customizable columns
- **Extensibility**: Supports any function with pandas DataFrame return

```python
class GeneralPipeline:
    def __init__(self, df_columns):
        self.evaluation_df = pd.DataFrame(columns=df_columns)
    
    def set_function(self, func):
        self.run_model_func = func
    
    def run_function(self, model_list, iterations, n_cores=1, **kwargs):
        # Parallel execution logic
```

#### EvaluationPipeline Class
- **Role**: Specialized for ML model evaluation
- **Features**: Cross-validation, feature selection integration, performance metrics
- **Output**: Structured evaluation results with metadata

### 3. Feature Selection System

#### Algorithm Categories
1. **Filter Methods**: Statistical relevance (F-regression, mutual info)
2. **Wrapper Methods**: Model-based selection (Greedy Forward, MRMR)
3. **Embedded Methods**: Regularization-based (Elastic Net, Lasso)
4. **Hybrid Methods**: ReliefF

#### MRMR Implementation
```python
def mrmr_select_fcq(X, y, K, verbose=0):
    # Maximum Relevance Minimum Redundancy
    # Balances feature relevance with inter-feature correlation
```

### 4. Machine Learning Integration

#### Supported Models
- **Regression**: Linear Regression, SVR, Random Forest, MLP, KNN
- **Classification**: Logistic Regression, SVC, Random Forest Classifier
- **Framework**: scikit-learn compatible interface

#### Hyperparameter Tuning
- Built-in support for model parameter optimization
- Cross-validation based evaluation
- Customizable scoring metrics

## Data Flow

### Standard Analysis Pipeline

1. **Data Loading**
   ```python
   # Load multi-omics data
   gdsc2 = load_drug_response_data()
   ccle = load_gene_expression_data()
   ```

2. **Data Integration**
   ```python
   # Join datasets by cell line
   joint_data = create_joint_dataset(gdsc2, ccle, drug='Palbociclib')
   ```

3. **Feature Engineering**
   ```python
   # Extract features and labels
   X, y = create_feature_and_label(joint_data, 'LN_IC50')
   ```

4. **Feature Selection**
   ```python
   # Apply MRMR or other algorithms
   selected_features, scores = mrmr_select_fcq(X, y, K=10)
   ```

5. **Model Training & Evaluation**
   ```python
   # Train models with selected features
   results = evaluate_models(X_selected, y, cv=5)
   ```

6. **Results Analysis**
   ```python
   # Performance assessment and feature importance
   analyze_results(results, selected_features)
   ```

### Advanced Workflow: Dynamic Feature Integration

For projects involving ODE-based dynamic modeling:

1. **ODE Simulation**
   ```python
   from UTIL_create_dynamic_features import create_dynamic_features
   dynamic_features = create_dynamic_features(params)
   ```

2. **Feature Combination**
   ```python
   # Combine static and dynamic features
   combined_features = combine_static_dynamic(static_X, dynamic_features)
   ```

3. **Comparative Analysis**
   ```python
   # Benchmark static vs dynamic features
   benchmark_results = compare_feature_types(combined_features, y)
   ```

## Parallel Processing Architecture

### Multi-core Execution

The pipeline system supports three execution modes:

1. **Serial** (`n_cores=1`): Debugging and small datasets
2. **Fixed Cores** (`n_cores=N`): Controlled parallel execution
3. **All Cores** (`n_cores=-1`): Maximum performance

```python
# Example: Run on 4 cores
pipeline.run_function(models, iterations=100, n_cores=4)

# Example: Use all available cores
pipeline.run_function(models, iterations=100, n_cores=-1)
```

### Joblib Integration

- **Backend**: Joblib Parallel for efficient parallel execution
- **Memory Management**: Automatic memory optimization
- **Progress Tracking**: Iteration-level progress reporting

## Configuration System

### Environment-based Configuration

#### Data Configuration (`data_config.env`)
```
DATA_PATH~user1 = '/path/to/data1'
DATA_PATH~user2 = '/path/to/data2'
```

#### User Configuration (`current_user.env`)
```
CURRENT_USER = user1
```

### Benefits
- **Multi-user Support**: Different researchers can use different data paths
- **Environment Isolation**: Development vs production configurations
- **Flexibility**: Easy path switching without code changes

## Extension Points

### Custom Feature Selection Algorithms

```python
def custom_feature_selector(X, y, K, **kwargs):
    # Implement your algorithm
    selected_features = your_algorithm(X, y, K)
    return selected_features, scores

# Integrate with existing framework
results = pipeline.run_with_custom_selector(custom_feature_selector)
```

### Custom Evaluation Metrics

```python
def custom_evaluation_metric(y_true, y_pred, **kwargs):
    # Define custom metric
    return custom_score

# Use in pipelines
pipeline.set_evaluation_metric(custom_evaluation_metric)
```

### Plugin Architecture

The modular design allows for easy integration of:
- New data sources
- Additional feature selection algorithms
- Custom ML models
- Specialized visualization tools

## Performance Considerations

### Memory Optimization

1. **Use Memory-optimized Scripts**: `*_memory_optimized.py` versions
2. **Batch Processing**: Process data in chunks for large datasets
3. **Selective Loading**: Load only required data subsets

### Computational Efficiency

1. **Parallelization**: Leverage multi-core processing
2. **Caching**: Use pickle caching for repeated operations
3. **Algorithm Selection**: Choose appropriate feature selection methods based on data size

## Testing and Validation

### Test Structure
- **Unit Tests**: `test_*.py` files for individual components
- **Integration Tests**: Pipeline-level validation
- **Benchmarking**: Performance comparison tests

### Validation Patterns
```python
# Example validation test
def test_feature_selection_consistency():
    # Ensure algorithm produces consistent results
    features1, scores1 = mrmr_select_fcq(X, y, K=10)
    features2, scores2 = mrmr_select_fcq(X, y, K=10)
    assert features1 == features2  # Should be deterministic
```

## Future Architecture Directions

### Planned Enhancements
1. **Distributed Computing**: Support for cluster environments
2. **GPU Acceleration**: CUDA-enabled feature selection and modeling
3. **Web Interface**: Browser-based analysis platform
4. **Real-time Analysis**: Streaming data processing capabilities

### Compatibility Roadmap
- **Python 3.11+** support
- **scikit-learn 1.3+** compatibility
- **Dask integration** for larger-than-memory datasets

---

*This architecture supports scalable biomarker discovery research with flexibility for custom extensions and optimizations.*
