import math
from matplotlib import pyplot as plt

def subplots(n_plots, sharex=True, sharey=True, **kwargs):
    """
    Create nicely sized and laid-out subplots for a desired number of plots.
    """
    # essentially we want to make the subplots as square as possible
    # number of rows is the largest factor of n_plots less than sqrt(n_plots)
    options = range(1, int(math.sqrt(n_plots)))
    n_rows = max(filter(lambda n: n_plots % n == 0, options))
    n_cols = int(n_plots / n_rows)
    # now generate the Figure and Axes pyplot objects
    SCALE_FACTOR = 3  # cosmetic scale factor to make larger plot
    figsize = (n_cols * SCALE_FACTOR, n_rows * SCALE_FACTOR)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=figsize,
        sharex=sharex, sharey=sharey,
        **kwargs
        )
    return fig, axs