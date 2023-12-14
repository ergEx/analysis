import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..utils import posterior_dist_2dplot


def hypothesis_fig(fig_dir):
    np.random.seed(0)
    cm = 1/2.54  # centimeters in inches (for plot size conversion)
    fig_size = (6.5 * cm , 5.75 * cm)

    colors_alpha = [np.array([0, 0, 1, 0.3], dtype=float), np.array([1, 0, 0, 0.3], dtype=float)]
    LIMITS = [-1,2]

    #EUT
    data = np.random.uniform(-3, 4, size = (2, 1000, 50, 2))
    data[:,:,:, 1] = data[:,:,:, 0] + np.random.normal(0,0.2, size = (2, 1000, 50))

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    fig, ax = posterior_dist_2dplot(fig, ax, data, colors_alpha, LIMITS, None)
    fig.savefig(os.path.join(fig_dir, 'EUT_pred.pdf'), dpi=600, bbox_inches='tight')

    #EE
    data = np.random.normal(0, 0.2, size = (2, 100,50, 2))
    data[:,:,:,1] = np.random.normal(1, 0.2, size = (2, 100,50))

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    fig, ax = posterior_dist_2dplot(fig, ax, data, colors_alpha, LIMITS, None)
    fig.savefig(os.path.join(fig_dir, f"EE_pred.pdf"), dpi=600, bbox_inches='tight')

    #EE2
    data = np.random.uniform(-3, 4, size = (2, 100,50, 2))
    data[:,:,:, 1] = data[:,:,:, 0] + np.abs(np.random.uniform(0, 4, size = (2, 100,50)))

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    fig, ax = posterior_dist_2dplot(fig, ax, data, colors_alpha, LIMITS, None)
    fig.savefig(os.path.join(fig_dir, f"EE2_pred.pdf"), dpi=600, bbox_inches='tight')