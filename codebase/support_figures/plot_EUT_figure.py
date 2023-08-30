import os

import matplotlib.pyplot as plt
import numpy as np
from ..utils import isoelastic_utility

plt.rcParams.update({
    "text.usetex": True})

cm = 1/2.54  # centimeters in inches (for plot size conversion)
fig_size = (13 * cm , (5.75*2) * cm)


def EUT_figure(fig_dir):
    etas = [-1.0,-0.5,0.0,0.5,1.0,1.5]
    cmap = plt.get_cmap("Dark2")
    colors = [cmap(i) for i in np.linspace(0, 1, len(etas))]

    x = np.linspace(2,1000,1000)

    fig_1, ax_1 = plt.subplots(1, 1, figsize=fig_size)
    for i, eta in  enumerate(etas):
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        u = isoelastic_utility(x,eta)
        rescaled_u = (u - u.min()) / (u.max() - u.min())
        ax_1.plot(x,rescaled_u, label=f'$\eta: {eta}$', color = colors[i])

        ax.plot(x,rescaled_u, label = f'$\eta: {eta}$', color = colors[i])
        ax.set(title='Isoelastic utility with different risk aversion parameters',
            xlabel=f'Wealth, $x$', ylabel=f'Scaled utility, $u(x;{eta})$')
        #legend = ax.legend(loc='upper left')
        if eta < 0:
            ax.set_title(f'Risk seeking, $\eta = {eta}$')
        elif eta == 0:
            ax.set_title(f'Risk neutral, $\eta = {eta}$')
        else:
            ax.set_title(f'Risk averse, $\eta = {eta}$')
        fig.savefig(os.path.join(fig_dir, f'utility_figure_eta_{eta}.png'), dpi=600, bbox_inches='tight')

    ax_1.set(title='Isoelastic utility with different risk aversion parameters',
            xlabel=f'Wealth, $x$',
            ylabel='Scaled utility, $u(x;\eta)$')
    legend = ax_1.legend(loc='lower right')
    legend.set_bbox_to_anchor((0.8, 0.))
    fig_1.savefig(os.path.join(fig_dir, 'utility_figure.png'), dpi=600, bbox_inches='tight')