# Module Reference Guide

## Core Framework Modules

### `dynmarker/` Package

#### `dynmarker.GeneralPipeline`
**Purpose**: Flexible execution framework for custom functions with parallel processing support.

**Key Features**:
- Parallel execution with joblib backend
- Customizable result DataFrame structure
- Function composition support

**Main Methods**:
```python
class GeneralPipeline:
    def __init__(self, df_columns: List[str])
    def set_function(self, func: Callable)
    def set_plotting_function(self, func: Callable)
    def run_function(self, model_list: List, iterations: int, n_cores: int = 1, **kwargs)
    def run_plot(self)
```

**Usage Example**:
```python
from dynmarker import GeneralPipeline

def custom_eval(model, verbose=0, **kwargs):
    return pd.DataFrame([{'score': 0.95}])

pipeline = GeneralPipeline(['score'])
pipeline.set_function(custom_eval)
pipeline.run_function([LinearRegression()], iterations=10, n_cores=4)
```

#### `dynmarker.EvaluationPipeline`
**Purpose**: Specialized pipeline for ML model evaluation with cross-validation.

**Key Features**:
- Built-in cross-validation support
- Feature selection integration
- Structured evaluation results

**Main Methods**:
```python
class EvaluationPipeline:
    def __init__(self)
    def set_function(self, func: Callable)
    def run_function(self, X, y, iterations: int, k_ranges: List[int], 
                    model_list: List, n_fold_splits: int = 5, n_cores: int = 1, **kwargs)
```

**Usage Example**:
```python
from dynmarker import EvaluationPipeline

pipeline = EvaluationPipeline()
pipeline.set_function(example_run_model_func)  # Predefined function
pipeline.run_function(X, y, iterations=5, k_ranges=[5, 10, 15], 
                     model_list=[LinearRegression(), RandomForestRegressor()])
```

#### `dynmarker.FeatureSelection`
**Purpose**: Comprehensive feature selection algorithm implementations.

**Key Algorithms**:
1. **MRMR (Maximum Relevance Minimum Redundancy)**
2. **Elastic Net Selection**
3. **ReliefF**
4. **Greedy Forward Selection**
5. **Filter Methods (F-regression, Mutual Info)**

**Main Functions**:
```python
# MRMR Feature Selection
def mrmr_select_fcq(X: pd.DataFrame, y: pd.Series, K: int, verbose: int = 0, 
                    return_index: bool = True) -> Tuple[List, List]

# Elastic Net Selection
def enet_select(X: pd.DataFrame, y: pd.Series, k: int, max_iter: int = 10000) -> Tuple[np.ndarray, np.ndarray]

# ReliefF Selection
def relieff_select(X: pd.DataFrame, y: pd.Series, k: int, n_jobs: int = 1) -> Tuple[np.ndarray, np.ndarray]

# Filter Methods
def filter_feature_selection(feature_data, label_data, method, K, get_selected_data=False)
```

**Usage Example**:
```python
from dynmarker.FeatureSelection import mrmr_select_fcq, enet_select

# MRMR selection
features, scores = mrmr_select_fcq(X, y, K=10, verbose=1)

# Elastic Net selection
indices, coefs = enet_select(X, y, k=10)
```

#### `dynmarker.DataLoader`
**Purpose**: Data loading and management utilities.

**Key Features**:
- Multi-source data integration
- Data validation and preprocessing
- Cache management

## Utility Modules

### `PathLoader.py`
**Purpose**: Flexible data path configuration management.

**Key Features**:
- Environment-based configuration
- Multi-user support
- Path resolution abstraction

**Main Class**:
```python
class PathLoader:
    def __init__(self, data_config_path: str, current_user_path: str)
    def get_data_path(self) -> str
    def set_user(self, user: str)
```

**Usage Example**:
```python
from PathLoader import PathLoader

path_loader = PathLoader('data_config.env', 'current_user.env')
data_path = path_loader.get_data_path()

# Load data using resolved path
with open(f'{data_path}data/dataset.pkl', 'rb') as f:
    data = pickle.load(f)
```

### `DataFunctions.py`
**Purpose**: Data processing and joint dataset creation.

**Key Functions**:
```python
def create_feature_and_label(df: pd.DataFrame, label_name: str = 'LN_IC50') -> Tuple[pd.DataFrame, pd.Series]
def create_joint_dataset_from_proteome_gdsc(drug_name: str, proteome_data, gdsc_data, drug_value: str) -> pd.DataFrame
```

**Usage Example**:
```python
from DataFunctions import create_feature_and_label, create_joint_dataset_from_proteome_gdsc

# Create joint dataset
joint_data = create_joint_dataset_from_proteome_gdsc("Palbociclib", proteome, gdsc, "LN_IC50")

# Split features and labels
X, y = create_feature_and_label(joint_data, 'LN_IC50')
```

### `toolkit.py`
**Purpose**: Pipeline utilities and helper functions.

**Key Functions**:
- Data transformation and imputation
- Hyperparameter tuning
- Model evaluation utilities

**Example Functions**:
```python
def transform_impute_by_zero_to_min_uniform(X_train, y_train)
def hypertune_svr(X, y, cv=5)
def select_preset_features(X, y, selected_features)
```

### `Visualisation.py`
**Purpose**: Data visualization and result plotting.

**Key Features**:
- Feature importance visualization
- Performance metric plots
- Comparative analysis charts

## Pipeline Scripts

### `SYPipelineScript.py`
**Purpose**: Comprehensive feature selection and modeling pipeline.

**Key Components**:
- SVR-based modeling with hyperparameter tuning
- Multi-stage feature selection (filter + wrapper)
- Consensus-based feature importance

**Main Functions**:
```python
def pipeline_func(X_train, y_train, **kwargs) -> Dict
def eval_func(X_test, y_test, pipeline_components=None, **kwargs) -> Dict
```

**Usage**:
```python
from SYPipelineScript import pipeline_func, eval_func

# Run the complete pipeline
results = pipeline_func(X_train, y_train)
evaluation = eval_func(X_test, y_test, pipeline_components=results)
```

### Benchmarking Scripts

#### `benchmark_models.py`
**Purpose**: Compare different ML models with various feature selection methods.

**Key Features**:
- Comprehensive model comparison
- Cross-validation evaluation
- Performance metric aggregation

#### `benchmark_feature_selection_batch.py`
**Purpose**: Batch benchmarking of feature selection algorithms.

**Key Features**:
- Multiple algorithm comparison
- Statistical significance testing
- Memory-optimized execution

#### Memory-optimized Variants
- `benchmark_*_memory_optimized.py`: Optimized for large datasets
- Reduced memory footprint through batch processing

## Data Processing Utilities

### `UTIL_create-dynamic-features.py`
**Purpose**: Generate dynamic features from ODE simulations.

**Key Features**:
- ODE parameter configuration
- Simulation time series generation
- Feature extraction from dynamics

### `UTIL_create-initial-conditions.py`
**Purpose**: Generate initial conditions for ODE simulations.

**Key Features**:
- Biological system parameterization
- State variable initialization
- Parameter sensitivity analysis

## Test Modules

### `test_*.py` Files
**Purpose**: Unit and integration tests for project components.

**Key Test Files**:
- `test_feature_importance_consensus.py`: Consensus algorithm validation
- `test_gene_subset_extraction.py`: Gene subset functionality tests
- `test_gene_subset_shap_sanity.py`: SHAP analysis validation

**Testing Pattern**:
```python
import unittest
from dynmarker.FeatureSelection import mrmr_select_fcq

class TestFeatureSelection(unittest.TestCase):
    def test_mrmr_consistency(self):
        # Test that MRMR produces consistent results
        features1, scores1 = mrmr_select_fcq(X, y, K=5)
        features2, scores2 = mrmr_select_fcq(X, y, K=5)
        self.assertEqual(features1, features2)
```

## Configuration Files

### `data_config.env`
**Format**:
```
DATA_PATH~user1 = '/path/to/data1'
DATA_PATH~user2 = '/path/to/data2'
```

**Purpose**: Multi-user data path configuration.

### `current_user.env`
**Format**:
```
CURRENT_USER = user1
```

**Purpose**: Current user selection for path resolution.

## Notebooks Directory Structure

### `thesis-notebooks/`
**Purpose**: Research analysis and result documentation.

**Key Subdirectories**:
- `result1/`: Initial benchmarking results
- `result2/`: Dynamic feature analysis
- `results3/`: Advanced analysis

### `notebooks/`
**Purpose**: General analysis and experimentation.

**Key Contents**:
- `visualise-consensus.ipynb`: Consensus visualization
- `ideas/`: Experimental approaches

### Project-specific Notebooks
- `project-phdndf/`: PhD research notebooks
- `project-drugnet/`: Drug network analysis
- `project-class/`: Classification model exploration

## Dependency Management

### `requirements.txt`
**Purpose**: Python package dependencies.

**Key Dependencies**:
- **ML Framework**: scikeras, skorch, xgboost
- **Feature Selection**: sklearn-relief, shap
- **Visualization**: seaborn, pyvis
- **Scientific Computing**: scipy, statsmodels, umap-learn
- **ODE Simulation**: libroadrunner, python-libsbml

### `pyproject.toml`
**Purpose**: Modern Python project configuration.

**Key Sections**:
- Project metadata (name, version, description)
- Python version requirements
- Dependency specifications

## File Naming Conventions

### Script Naming Patterns
- `AN_*.py`: Analysis scripts
- `SCRIPT_*.py`: Pipeline execution scripts
- `benchmark_*.py`: Performance benchmarking
- `UTIL_*.py`: Utility functions
- `test_*.py`: Test files

### Notebook Naming
- `*_analysis.ipynb`: Data analysis notebooks
- `*_pipeline.ipynb`: Pipeline experimentation
- `*_visualisation.ipynb`: Result visualization

## Module Interactions

### Data Flow Between Modules

```
PathLoader → DataLoader → DataFunctions → FeatureSelection → EvaluationPipeline
     ↓
Toolkit → Visualization → Results Storage
```

### Typical Import Patterns

```python
# Core framework
from dynmarker import GeneralPipeline, EvaluationPipeline
from dynmarker.FeatureSelection import mrmr_select_fcq, enet_select

# Data management
from PathLoader import PathLoader
from DataFunctions import create_feature_and_label

# Utilities
from toolkit import hypertune_svr, transform_impute_by_zero_to_min_uniform
```

## Extension Patterns

### Adding New Feature Selection Algorithms

1. **Implement Function** in `dynmarker/FeatureSelection.py`:
```python
def new_algorithm_select(X, y, K, **kwargs):
    # Implementation
    return selected_features, scores
```

2. **Integrate with Pipeline**:
```python
from dynmarker.FeatureSelection import new_algorithm_select

# Use in custom pipeline
features, scores = new_algorithm_select(X, y, K=10)
```

### Creating Custom Pipelines

1. **Define Pipeline Functions**:
```python
def custom_pipeline_func(X_train, y_train, **kwargs):
    # Custom logic
    return pipeline_components

def custom_eval_func(X_test, y_test, pipeline_components, **kwargs):
    # Evaluation logic
    return results
```

2. **Integrate with Framework**:
```python
from dynmarker import GeneralPipeline

pipeline = GeneralPipeline(['custom_metrics'])
pipeline.set_function(custom_pipeline_func)
```

## Performance Optimization

### Memory Management
- Use `*_memory_optimized.py` scripts for large datasets
- Implement batch processing for data chunks
- Utilize selective data loading

### Computational Efficiency
- Leverage parallel processing (`n_cores` parameter)
- Use appropriate feature selection algorithms for data size
- Implement caching for repeated operations

---

*This module reference provides a comprehensive overview of all project components and their interactions. For specific implementation details, refer to the source code and inline documentation.*
