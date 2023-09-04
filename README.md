# ode-biomarker-project: dynmarker

Dynmarker (Dynamic biomarker) utilises dynamic modelling via ordinary differential equations (ODEs) and machine learning (ML) to identify biomarkers of drug response in cancer. 

## Instructions 

a data repository is required to load the data which can be used to reproduce the results in the manuscript. The data repository can be found at: ()[link-here!]

### Data Installation

After downloading the data repository, the recommended way of loading data for the project is by using the `data_configs.env` file. 

data config file format:
DATA_PATH~user1 = 'path1'
DATA_PATH~user2 = 'path2'
... 

current user file format:
CURRENT_USER = user1

set the `DATA_PATH` variable to the path of the data repository, and set the `CURRENT_USER` variable to the user you are currently using. Any username can be used, as long as it is consistent with the user name in the data config file.

In any .py file, the data can be loaded by using the following code:

```python
from PathLoader import PathLoader
path_loader = PathLoader('data_config.env', 'current_user.env')
data_path = path_loader.get_data_path()
```

