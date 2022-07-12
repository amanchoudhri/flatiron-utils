import math

from typing import List, Tuple


import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def subplots(n_plots, scale_factor=4, sharex=True, sharey=True, **kwargs) -> Tuple[Figure, List[Axes]]:
    """
    Create nicely sized and laid-out subplots for a desired number of plots.
    """
    # essentially we want to make the subplots as square as possible
    # number of rows is the largest factor of n_plots less than sqrt(n_plots)
    options = range(1, int(math.sqrt(n_plots) + 1))
    n_rows = max(filter(lambda n: n_plots % n == 0, options))
    n_cols = int(n_plots / n_rows)
    # now generate the Figure and Axes pyplot objects
    # cosmetic scale factor to make larger plot
    figsize = (n_cols * scale_factor, n_rows * scale_factor)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=figsize,
        sharex=sharex, sharey=sharey,
        **kwargs
        )
    flattened_axes = [ax for ax_row in axs for ax in ax_row]
    return fig, flattened_axes

def plot_calibration_curve(
    bin_edges,
    true_props,
    ax = None
    ) -> Tuple[np.ndarray, np.ndarray, Axes]:
    """
    Plot the given calibration curve, returning the x and y values
    along with the axis on which the curve was plotted.
    """
    if not ax:
        _, ax = plt.subplots()
    ax.scatter(bin_edges, true_props)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='dashed')
    ax.set_xlabel('Probability assigned to region around mode')
    ax.set_ylabel('True proportion of samples in region')
    return bin_edges, true_props, ax
