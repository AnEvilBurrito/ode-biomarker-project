# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# math and metrics
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

def plot_box_plot(df, category_name, score_name, title, x_label, y_label, 
                                      fontsize=18, minitext_size=12, tick_fontsize=18,
                                      ax=None,
                                      plot_jitter=True,
                                      show_correlation_coefficient=True, 
                                      palette='Set2',
                                      **kwargs):
    
    # palette can also be in a seaborn palette object: palette=sns.color_palette(['#e41a1c', '#377eb8', '#4daf4a'], n_colors=3)

    if ax is None:
        ax = plt.gca()

    # using seaborn boxplot and stripplot
    sns.boxplot(x=category_name, y=score_name, data=df, ax=ax, width=0.3,
                palette=palette,
                boxprops=dict(linewidth=1, alpha=0.25),
                medianprops=dict(color="black", alpha=1), **kwargs)
    if plot_jitter:
        sns.stripplot(x=category_name, y=score_name, data=df, ax=ax, alpha=0.5, size=10, palette=palette, hue=category_name)

    # set the title and labels
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)

    # set the mini-labels font size to large
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    return ax

def plot_predictions_vs_actual_values(y_test, y_pred, title='', x_label='Actual', y_label='Predicted', 
                                      fontsize=18, minitext_size=12, tick_fontsize=18,
                                      ax=None,
                                      plot_line_of_y_equals_x=True,
                                      plot_trend_line=True,
                                      show_correlation_coefficient=True, *args, **kwargs):
    
    '''
    plot the predictions vs actual values, the parameter data types accepts the following:
    y_test: array-like, shape (n_samples,)
        Ground truth (correct) target values.
    y_pred: array-like, shape (n_samples,)
        Estimated target values.
    title: string
        The title of the plot. 
    x_label: string
        The x-axis label of the plot.
    y_label: string
        The y-axis label of the plot.
    fontsize: int
        The font size of the title, x_label and y_label.
    minitext_size: int
        The font size of the correlation coefficient and p-value.
    tick_fontsize: int
        The font size of the x and y ticks.
    ax: matplotlib.axes.Axes
        The axes to plot on. If None, the current axes will be used.
    plot_line_of_y_equals_x: bool
        Whether to plot the line of y=x.
    plot_trend_line: bool
        Whether to plot a trend line.
    show_correlation_coefficient: bool
        Whether to show the correlation coefficient and p-value.
    **kwargs: dict
        Other keyword arguments are passed to matplotlib.axes.Axes.scatter.
    returns matplotlib.axes.Axes object, this can be used to add more plots to the same figure.
    '''
    
    if ax is None:
        ax = plt.gca()
    
    if plot_line_of_y_equals_x:
        ax.plot(y_test, y_test, linestyle='--', color='black', linewidth=1)

    # title
    ax.set_title(title, fontsize=fontsize)
    ax.scatter(y_test, y_pred, *args, **kwargs)

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

    ax.set_aspect('auto')
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    # change x and y tick size to big font
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid()
    
    return ax

def plot_correlation(df, x_field, y_field, ax,
                     title='', xlabel='', ylabel='',
                      **kwargs):
    
    '''
    Inputs: 
    df: pandas dataframe
    x_field: string
        The name of the column in the dataframe to be used as the x-axis.
    y_field: string
        The name of the column in the dataframe to be used as the y-axis.
    ax: matplotlib.axes.Axes
        The axes to plot on. If None, the current axes will be used.
    title: string
        The title of the plot.
    xlabel: string
        The x-axis label of the plot.
    ylabel: string
        The y-axis label of the plot.

    returns matplotlib.axes.Axes object, this can be used to add more plots to the same figure. 

    '''
    """Plot a scatter plot with a line of best fit and correlation coefficient"""
    # plot the scatter plot
    ax.scatter(df[x_field], df[y_field], **kwargs)
    # create line of best fit
    try: 
        m, b = np.polyfit(df[x_field], df[y_field], 1)
    except Exception as e:
        m, b = 0, 0
    ax.plot(df[x_field], m*df[x_field] + b, color='grey', linestyle='--')
    # show correlation coefficient in the title
    # corr = str(round(np.corrcoef(df[x_field], df[y_field])[0, 1], 2))

    corr, pval = pearsonr(df[x_field], df[y_field])
    corr = str(round(corr, 2))
    if pval < 0.05:
        corr = corr + '*'
    else: 
        corr = corr + 'n.s.'
    if title == '':
        ax.set_title(f'{x_field} vs {y_field} (r={corr})')
    else:
        ax.set_title(title)
    
    if xlabel == '':
        ax.set_xlabel(x_field)
    else:
        ax.set_xlabel(xlabel)
    
    if ylabel == '':
        ax.set_ylabel(y_field)
    else:
        ax.set_ylabel(ylabel)
    # add corr as a caption in the plot
    ax.text(0.05, 0.97, f'r={corr}', transform=ax.transAxes, fontsize=14, verticalalignment='top')

    return ax


def plot_histogram(df, field, ax, title='', xlabel='', ylabel='', **kwargs):
    """Plot a histogram"""
    ax.hist(df[field], **kwargs)
    if title == '':
        ax.set_title(field)
    else:
        ax.set_title(title)
    if xlabel == '':
        ax.set_xlabel(field)
    else:
        ax.set_xlabel(xlabel)
    if ylabel == '':
        ax.set_ylabel('Count')
    else:
        ax.set_ylabel(ylabel)
    return ax