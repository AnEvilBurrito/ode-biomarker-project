# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# math and metrics
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

def plot_predictions_vs_actual_values(y_test, y_pred, title, x_label, y_label, 
                                      fontsize=18, minitext_size=12, tick_fontsize=18,
                                      ax=None,
                                      plot_line_of_y_equals_x=True,
                                      plot_trend_line=True,
                                      show_correlation_coefficient=True):
    
    if ax is None:
        ax = plt.gca()
    
    if plot_line_of_y_equals_x:
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='k')

    # title
    ax.set_title(title, fontsize=fontsize)
    ax.scatter(y_test, y_pred)

    # show correlation coefficient and p-value and r_squared
    if show_correlation_coefficient:
        corr, p_val = pearsonr(y_test, y_pred)
        r_squared = r2_score(y_test, y_pred)
        ax.text(0.1, 0.9, f'r={corr:.2f}, p-value: {p_val:.2f}', fontsize=minitext_size, transform=ax.transAxes)
        ax.text(0.1, 0.95, f'r-squared={r_squared:.2f}', fontsize=minitext_size, transform=ax.transAxes)

    # plot a trend line
    if plot_trend_line:
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_test, p(y_test), 'grey')

    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    # change x and y tick size to big font
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid()
    
    return ax