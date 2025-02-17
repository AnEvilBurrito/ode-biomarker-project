import os
import sys

path = os.getcwd()
# find the string 'project' in the path, return index
index_project = path.find('project')
# slice the path from the index of 'project' to the end
project_path = path[:index_project+7]
# set the working directory
sys.path.append(project_path)

from tqdm import tqdm
from toolkit import *

'''
Overarching Pipeline Function 
'''




'''
Powerkit Pipeline Functions - Pipeline and Evaluation
'''

def pipeline_tree_methods():
    pass 


def eval_standard():
    pass