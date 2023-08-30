import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_hypotheses(data):
    cm = 1/2.54
    fig_size = (6.5 * cm , 5.75 * cm)
    plt.rcParams.update({"text.usetex": True})
    sns.set_context('paper', font_scale=1.1)

    fiducials = [0,1]
    limits = [-0.5, 1.5]

    fig, ax = plt.subplots()

    sns.kdeplot(data[:, 0], data[:, 1], cmap="YlOrBr", shade=True, ax=ax)
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

    #h0
    data = np.random.uniform(-3, 4, (10000, 2))
    data[:, 1] = data[:, 0] + np.random.normal(0,0.2,10000)

    fig, ax = plot_hypotheses(data)
    fig.savefig(os.path.join(fig_dir, f"h_0.pdf"), dpi=600, bbox_inches='tight')

    #h1
    data = np.random.normal(0, 0.2, (10000, 2))
    data[:,1] = np.random.normal(1, 0.2, (10000))

    fig, ax = plot_hypotheses(data)
    fig.savefig(os.path.join(fig_dir, f"h_1.pdf"), dpi=600, bbox_inches='tight')

    #h2
    data = np.random.uniform(-3, 4, (10000, 2))
    data[:, 1] = data[:, 0] + np.abs(np.random.uniform(0, 4, (10000)))

    fig, ax = plot_hypotheses(data)
    fig.savefig(os.path.join(fig_dir, f"h_2.pdf"), dpi=600, bbox_inches='tight')