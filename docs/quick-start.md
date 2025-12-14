# Quick Start Guide

## Installation (5-minute setup)

### 1. Clone and Setup Environment
```bash
git clone https://github.com/AnEvilBurrito/ode-biomarker-project.git
cd ode-biomarker-project
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### 2. Basic Data Configuration
Create minimal configuration files:

**data_config.env**
```
DATA_PATH~default = '/path/to/your/data'
```

**current_user.env**
```
CURRENT_USER = default
```

### 3. Verify Installation
```python
# quick_test.py
from dynmarker import GeneralPipeline
from dynmarker.FeatureSelection import naive_test_regression
from PathLoader import PathLoader

print("✅ Installation successful!")
print("✅ Basic imports working!")
```

## First Analysis (10-minute tutorial)

### Sample Workflow
```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from dynmarker.FeatureSelection import naive_test_regression

# Create sample data (replace with your actual data)
X, y = make_regression(n_samples=100, n_features=50, noise=0.1)
feature_data = pd.DataFrame(X)
label_data = pd.Series(y)

# Quick evaluation
results = naive_test_regression(feature_data, label_data, cv=3, verbose=1)
print("Basic ML models tested successfully!")
```

### Simple Feature Selection
```python
from dynmarker.FeatureSelection import mrmr_select_fcq

# MRMR feature selection on sample data
selected_features, scores = mrmr_select_fcq(
    pd.DataFrame(X), 
    pd.Series(y), 
    K=10, 
    verbose=1
)
print(f"Selected top 10 features: {selected_features}")
```

## Common Tasks

### Load and Join Data
```python
from PathLoader import PathLoader
from DataFunctions import create_feature_and_label
import pickle

path_loader = PathLoader('data_config.env', 'current_user.env')

# Example data loading pattern
# with open(f'{path_loader.get_data_path()}data/your_dataset.pkl', 'rb') as f:
#     your_data = pickle.load(f)

# Create feature/label split
# feature_data, label_data = create_feature_and_label(your_data, 'target_column')
```

### Run Benchmark
```python
# Sample benchmarking pattern
from benchmark_models import run_benchmark

# results = run_benchmark(
#     feature_data, 
#     label_data,
#     feature_selection_methods=['mrmr'],
#     models=['linear', 'rf'],
#     cv_folds=3,
#     n_iterations=5
# )
```

### Custom Pipeline
```python
from dynmarker import GeneralPipeline
from sklearn.linear_model import LinearRegression

def simple_model_test(model, verbose=0, **kwargs):
    # Your custom evaluation logic
    results = {'score': 0.85, 'model_name': model.__class__.__name__}
    return pd.DataFrame([results])

pipeline = GeneralPipeline(['score', 'model_name'])
pipeline.set_function(simple_model_test)

models = [LinearRegression()]
pipeline.run_function(models, iterations=3, n_cores=1)
print(pipeline.evaluation_df)
```

## Troubleshooting

### Quick Checks
1. **Environment**: `python --version` (should be >=3.10)
2. **Imports**: Run verification script above
3. **Data Path**: Check `data_config.env` and `current_user.env` files exist

### Common Solutions
- **Import errors**: Re-run `uv pip install -e .`
- **Data path issues**: Verify file paths in configuration files
- **Memory issues**: Use `n_cores=1` for debugging

## Next Steps
- Explore `thesis-notebooks/` for research examples
- Try `SYPipelineScript.py` for full analysis pipeline
- Review `docs/onboarding.md` for comprehensive documentation

---

*This quick start guide gets you running in under 15 minutes. For detailed documentation, see the full onboarding guide.*
