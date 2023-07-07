import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_hypotheses(data):
    cm = 1/2.54
    fig_size = (7.5 * cm , 6 * cm)
    fiducials = [0,1]
    limits = [-1, 2]

    fig, ax = plt.subplots()

    sns.kdeplot(data[:, 0], data[:, 1], cmap="YlOrBr", shade=True, ax=ax)

    ax.set(xlim=limits, ylim=limits, xlabel='Additive condition', ylabel='Multiplicative condition',
           xticks=np.linspace(limits[0], limits[1], limits[1]-limits[0]+1),
           yticks=np.linspace(limits[0], limits[1], limits[1]-limits[0]+1))
    sns.lineplot(x=limits, y=limits, color='black', alpha=0.5, linestyle='--', ax=ax)

    ax.axvline(fiducials[0], color='blue', alpha=0.5, linestyle='--')
    ax.axhline(fiducials[1], color='red', alpha=0.5, linestyle='--')

    fig.set_size_inches(fig_size)

    return fig, ax

np.random.seed(0)
fig_dir = os.path.join('figs')

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