import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..utils import posterior_dist_2dplot


def plot_hypotheses(data):
    cm = 1/2.54
    fig_size = (6.5 * cm , 5.75 * cm)
    plt.rcParams.update({"text.usetex": True})
    sns.set_context('paper', font_scale=1.1)

    fiducials = [0,1]
    limits = [-0.5, 1.5]

    fig, ax = plt.subplots()

    sns.kdeplot(data[:, 0], data[:, 1], cmap="coolwarm", shade=True, ax=ax)
    ticks = np.arange(limits[0], limits[1] + 0.5, 0.5)

    ax.set(xlim=limits, ylim=limits, xlabel="$\eta^{\mathrm{add}}$", ylabel='$\eta^{\mathrm{mul}}$',
           xticks=ticks,
           yticks=ticks)
    sns.lineplot(x=limits, y=limits, color='black', alpha=0.5, linestyle='--', ax=ax)

    ax.axvline(fiducials[0], color='blue', alpha=0.5, linestyle='--')
    ax.axhline(fiducials[1], color='red', alpha=0.5, linestyle='--')

    fig.set_size_inches(fig_size[1], fig_size[1])

    return fig, ax


def hypothesis_fig(fig_dir):
    np.random.seed(0)

    colors_alpha = [np.array([0, 0, 1, 0.3], dtype=float), np.array([1, 0, 0, 0.3], dtype=float)]
    LIMITS = [-1,2]

    #EUT
    data = np.random.uniform(-3, 4, size = (2, 1000, 50, 2))
    data[:,:,:, 1] = data[:,:,:, 0] + np.random.normal(0,0.2, size = (2, 1000, 50))

    fig, ax = posterior_dist_2dplot(fig, ax, data, colors_alpha, LIMITS, None)
    fig.savefig(os.path.join(fig_dir, 'EUT_pred.pdf'), dpi=600, bbox_inches='tight')

    #EE
    data = np.random.normal(0, 0.2, size = (2, 100,50, 2))
    data[:,:,:,1] = np.random.normal(1, 0.2, size = (2, 100,50))

    fig, ax = posterior_dist_2dplot(fig, ax, data, colors_alpha, LIMITS, None)
    fig.savefig(os.path.join(fig_dir, f"EE_pred.pdf"), dpi=600, bbox_inches='tight')

    #EE2
    data = np.random.uniform(-3, 4, size = (2, 100,50, 2))
    data[:,:,:, 1] = data[:,:,:, 0] + np.abs(np.random.uniform(0, 4, size = (2, 100,50)))

    fig, ax = posterior_dist_2dplot(fig, ax, data, colors_alpha, LIMITS, None)
    fig.savefig(os.path.join(fig_dir, f"EE2_pred.pdf"), dpi=600, bbox_inches='tight')