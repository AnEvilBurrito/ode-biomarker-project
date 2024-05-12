import numpy as np
import pandas as pd 

def get_dynamic_features(col_data: pd.Series, 
                         normalise: bool = True,
                         abs_change_tolerance: float = 0.01) -> list:
    # dynamic features
    auc = np.trapz(col_data)
    max_val = np.max(col_data)
    max_time = np.argmax(col_data)
    min_val = np.min(col_data)
    min_time = np.argmin(col_data)

    median_val = np.median(col_data)

    # calculation of total fold change (tfc)
    start_val = col_data.iloc[0]
    end_val = col_data.iloc[-1]

    tfc = 0 
    if start_val == 0:
        tfc = 1000
    else:
        if end_val - start_val >= 0:
            tfc = (end_val - start_val) / start_val
        elif end_val - start_val < 0:
            if end_val == 0:
                tfc = -1000
            else:
                tfc = -((start_val - end_val) / end_val)

    # calculation of time to stability (tsv)
    tsv = len(col_data)
    while tsv > 1:
        if abs(col_data.iloc[tsv-1] - col_data.iloc[tsv-2]) < abs_change_tolerance:
            tsv -= 1
        else:
            tsv_value = col_data.iloc[tsv-1]
            break
    if tsv == 1:
        tsv_value = col_data.iloc[0]

    max_sim_time = len(col_data)
    n_auc = auc / max_sim_time
    n_max_time = max_time / max_sim_time
    n_min_time = min_time / max_sim_time
    n_tsv = tsv / max_sim_time
    
    if not normalise:
        # reset the values to the original values
        n_auc = auc
        n_max_time = max_time
        n_min_time = min_time
        n_tsv = tsv 

    dynamic_features = [n_auc, median_val, tfc, n_max_time,
                        max_val, n_min_time, min_val, n_tsv, tsv_value, start_val]

    return dynamic_features
