# Dynmarker AI Onboarding Document

## Overview

Welcome to the **ode-biomarker-project**, also known as **dynmarker** (Dynamic biomarker). This project utilizes dynamic modeling via ordinary differential equations (ODEs) and machine learning (ML) to identify biomarkers of drug response in cancer.

### Key Features

- **Dynamic Modeling**: ODE-based simulation of biological systems
- **Multi-Omics Integration**: Combines gene expression, proteomic, and drug response data
- **Advanced Feature Selection**: MRMR, ReliefF, elastic net, and wrapper methods
- **Modular Pipeline Architecture**: Flexible evaluation and benchmarking frameworks
- **Parallel Processing**: Efficient computation across multiple cores
- **Comprehensive Benchmarking**: Comparison of feature selection and modeling approaches

## System Requirements

### Software Prerequisites

- **Python**: >= 3.10
- **Package Manager**: pip or uv
- **Data Repository**: External data repository required (see Data Setup section)

### Hardware Recommendations

- **Memory**: 8GB+ RAM recommended for large datasets
- **CPU**: Multi-core processor for parallel processing
- **Storage**: Sufficient space for datasets and results (varies by data repository size)

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/AnEvilBurrito/ode-biomarker-project.git
cd ode-biomarker-project
```

### 2. Create Python Environment

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Using uv
uv pip install -e .

# Or using pip
pip install -e .
```

### 4. Data Repository Setup

The project requires an external data repository. After obtaining the data:

1. Configure `data_config.env`:
```
DATA_PATH~user1 = '/path/to/data/repository'
DATA_PATH~user2 = '/path/to/alternate/data/repository'
```

2. Configure `current_user.env`:
```
CURRENT_USER = user1
```

### 5. Verify Installation

```python
from dynmarker import GeneralPipeline, EvaluationPipeline
from dynmarker.FeatureSelection import mrmr_select_fcq
import PathLoader

print("Installation successful!")
```

## Project Architecture

### Core Modules

#### `dynmarker/` - Main Package
- **`dynmarker.py`**: Core functionality (placeholder - functionality distributed across other modules)
- **`GeneralPipeline.py`**: Flexible pipeline for custom functions
- **`EvaluationPipeline.py`**: ML model evaluation framework
- **`FeatureSelection.py`**: Comprehensive feature selection algorithms
- **`DataLoader.py`**: Data loading utilities

#### Utility Modules
- **`PathLoader.py`**: Flexible data path management
- **`DataFunctions.py`**: Data processing and joint dataset creation
- **`toolkit.py`**: Pipeline utilities and functions
- **`Visualisation.py`**: Visualization tools

### Data Flow

1. **Data Loading**: PathLoader → DataLoader → DataFunctions
2. **Preprocessing**: Feature extraction and transformation
3. **Feature Selection**: Multiple algorithm options
4. **Model Training**: ML pipeline execution
5. **Evaluation**: Performance assessment and benchmarking
6. **Results Storage**: Pickle files and analysis notebooks

## Getting Started

### Basic Workflow Example

```python
from PathLoader import PathLoader
from DataFunctions import create_feature_and_label
from dynmarker.FeatureSelection import naive_test_regression

# Load data
path_loader = PathLoader('data_config.env', 'current_user.env')
data_path = path_loader.get_data_path()

# Example: Load and join datasets (replace with actual data loading)
# data_df = load_your_dataset_function(...)
# feature_data, label_data = create_feature_and_label(data_df)

# Quick test
# results = naive_test_regression(feature_data, label_data, cv=5, verbose=1)
```

### Using the General Pipeline

```python
from dynmarker import GeneralPipeline
import pandas as pd
import numpy as np

def custom_evaluation_function(model, verbose=0, **kwargs):
    # Your custom function logic
    results = {'performance': 0.95, 'features': [1, 2, 3]}
    return pd.DataFrame([results])

# Setup pipeline
pipeline = GeneralPipeline(['performance', 'features'])
pipeline.set_function(custom_evaluation_function)

# Run with multiple models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

models = [LinearRegression(), RandomForestRegressor()]
pipeline.run_function(models, iterations=10, n_cores=4)

print(pipeline.evaluation_df)
```

## Data Management

### Supported Data Types

1. **Drug Response**: GDSC2 database
2. **Gene Expression**: CCLE Public 22Q2
3. **Proteomic Data**: Various proteomic datasets
4. **Protein Interactions**: STRING database
5. **Dynamic Features**: ODE simulation outputs

### Data Configuration System

The `PathLoader` system provides flexible data path management:

```python
from PathLoader import PathLoader

# Initialize with configuration files
path_loader = PathLoader('data_config.env', 'current_user.env')

# Get current data path
data_path = path_loader.get_data_path()

# Load data using the path
import pickle
with open(f'{data_path}data/drug-response/GDSC2/cache_gdsc2.pkl', 'rb') as f:
    gdsc2 = pickle.load(f)
```

### Pre-processing Guidelines

Refer to `pre-processing-guideline.md` for detailed data preparation standards covering:
- Data transformation methods
- Missing value imputation
- Dataset integration patterns
- Identifier mapping strategies

## Feature Selection Algorithms

### Available Methods

1. **Filter Methods**:
   - F-regression (Pearson correlation)
   - Mutual information
   - SelectKBest wrapper

2. **Wrapper Methods**:
   - Greedy Forward Selection
   - MRMR (Maximum Relevance Minimum Redundancy)

3. **Embedded Methods**:
   - Elastic Net
   - Lasso

4. **Hybrid Methods**:
   - ReliefF

### Example Usage

```python
from dynmarker.FeatureSelection import mrmr_select_fcq, enet_select

# MRMR feature selection
selected_features, scores = mrmr_select_fcq(X, y, K=10, verbose=1)

# Elastic Net selection
indices, coefficients = enet_select(X, y, k=10)
```

## Pipeline Scripts

The project includes several ready-to-use pipeline scripts:

### SYPipelineScript
- Comprehensive feature selection and modeling pipeline
- SVR-based modeling with hyperparameter tuning
- Consensus-based feature importance

### Benchmarking Scripts
- `benchmark_*.py`: Various benchmarking configurations
- Feature selection algorithm comparisons
- Memory-optimized versions available

## Advanced Usage

### Custom Pipeline Development

```python
from dynmarker import GeneralPipeline

class CustomPipeline:
    def __init__(self):
        self.pipeline = GeneralPipeline(['custom_metrics'])
    
    def my_pipeline_func(self, model, **kwargs):
        # Custom pipeline logic
        results = self.run_custom_analysis(model, kwargs)
        return pd.DataFrame([results])
    
    def run_analysis(self, models, params):
        self.pipeline.set_function(self.my_pipeline_func)
        self.pipeline.run_function(models, **params)
```

### Parallel Processing Configuration

```python
# Single core (debugging)
pipeline.run_function(models, iterations=10, n_cores=1)

# All available cores
pipeline.run_function(models, iterations=10, n_cores=-1)

# Specific number of cores
pipeline.run_function(models, iterations=10, n_cores=4)
```

### Dynamic Feature Integration

The project supports integration of dynamic features from ODE simulations:

```python
from UTIL_create_dynamic_features import create_dynamic_features

# Generate dynamic features from biological models
dynamic_features = create_dynamic_features(
    initial_conditions, 
    ode_parameters, 
    simulation_time
)
```

## Benchmarking & Evaluation

### Performance Metrics

- **Regression**: Mean Squared Error, Pearson correlation
- **Classification**: Accuracy, F1-score
- **Feature Selection**: Stability, relevance scores

### Benchmarking Framework

```python
from benchmark_models import run_benchmark

# Run comprehensive benchmark
results = run_benchmark(
    feature_data, 
    label_data, 
    feature_selection_methods=['mrmr', 'relieff', 'enet'],
    models=['linear', 'svr', 'rf'],
    cv_folds=5,
    n_iterations=10
)
```

## Troubleshooting

### Common Issues

1. **Data Path Errors**: Ensure `data_config.env` and `current_user.env` are properly configured
2. **Memory Issues**: Use memory-optimized scripts for large datasets
3. **Dependency Conflicts**: Use the provided `requirements.txt` or `pyproject.toml`

### Debugging Tools

- `debug_protein_mapping.py`: Protein identifier mapping diagnostics
- Diagnostic notebooks in `thesis-notebooks/`
- Benchmark diagnostic scripts

## Development Guidelines

### Code Organization

- **Modules**: Functional grouping in `dynmarker/` package
- **Scripts**: Standalone executable scripts in root directory
- **Notebooks**: Research and analysis notebooks in dedicated folders
- **Tests**: Test files prefixed with `test_`

### Contribution Standards

1. **Documentation**: Include docstrings and update relevant documentation
2. **Testing**: Add tests for new functionality
3. **Code Style**: Follow existing project conventions
4. **Data Handling**: Adhere to pre-processing guidelines

## Research Workflows

### Typical Analysis Pipeline

1. **Data Preparation**: Load and preprocess multi-omics data
2. **Feature Engineering**: Generate dynamic features if applicable
3. **Feature Selection**: Apply MRMR or other algorithms
4. **Model Training**: Train ML models with selected features
5. **Evaluation**: Assess performance and feature importance
6. **Benchmarking**: Compare against baseline methods
7. **Visualization**: Create summary plots and reports

### Example Research Questions

- Which features best predict drug response for CDK4/6 inhibitors?
- How do dynamic features compare to static expression features?
- What is the optimal feature selection method for proteomic data?

## Next Steps

1. **Explore Notebooks**: Review `thesis-notebooks/` for research examples
2. **Run Basic Pipeline**: Try `SYPipelineScript.py` with sample data
3. **Customize Analysis**: Modify existing scripts for your research questions
4. **Benchmark Methods**: Use benchmarking scripts to compare approaches

## Support & Resources

- **Documentation**: This onboarding guide and code docstrings
- **Example Scripts**: Pipeline scripts in root directory
- **Research Notebooks**: Analysis examples in various folders
- **Issue Tracking**: GitHub repository issues

For specific questions or contributions, refer to the project's GitHub repository and documentation.

---

*Last Updated: December 2025*  
*Project Version: 0.1.0*  
*Compatible with Python >= 3.10*
