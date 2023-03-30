import pandas as pd
import numpy as np

def load_ccle(use_cache=True):
    if use_cache:
        try:
            return pd.read_pickle('ccle.pkl')
        except FileNotFoundError:
            pass