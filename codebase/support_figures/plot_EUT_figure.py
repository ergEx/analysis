import os

import matplotlib.pyplot as plt
import numpy as np

from ..plotting_functions import draw_brace
from ..utils import isoelastic_utility

plt.rcParams.update({
    "text.usetex": True})

cm = 1/2.54  # centimeters in inches (for plot size conversion)
fig_size = (13 * cm , (5.75*2) * cm)


def EUT_figure(fig_dir):
    x_values = np.linspace(1, 1000, 1000)
    vertical_lines_x = [200, 500, 800]
    text = ['Tails','No wealth change', 'Heads']
    colors = ['green','orange']

    #eta specific values (if other values of eta is plotted these needs to be changed manually)
    ypos = [1500,4.5]
    fac = [1.03,1.017]

    for j, eta in enumerate([-0.5,0.5]):

        fig, ax = plt.subplots(1,1, figsize = (10,10))
        ax.plot(x_values, isoelastic_utility(x_values,eta), label=r'$f(x) = x$', color = colors[j])

        for i, line_x in enumerate(vertical_lines_x):
            ax.vlines(x=line_x, ymin=ypos[j], ymax=isoelastic_utility(line_x,eta), color='grey')
            ax.hlines(y = isoelastic_utility(line_x,eta), color='grey', xmin = 0, xmax = line_x)
            ax.scatter(x = line_x, y = isoelastic_utility(line_x,eta), color = 'grey')
            ax.text(line_x,0, text[i], ha='center', va='bottom')


        ax.set(ylabel = r"$u(x)$", xlabel = r"$x$", xticks = vertical_lines_x)

        draw_brace(ax = ax,
                span = (vertical_lines_x[0], vertical_lines_x[2]),
                pos = 0,
                text = '',
                col = colors[j],
                orientation = 'horizontal')

        draw_brace(ax = ax,
                span = (isoelastic_utility(vertical_lines_x[0],eta), isoelastic_utility(vertical_lines_x[2],eta)),
                pos = 0.5,
                text = 'Average utility accepting gamble',
                col = colors[j],
                orientation = 'vertical')

        ax.text(75, isoelastic_utility(vertical_lines_x[1],eta)*fac[j], 'Average utility not accepting gamble', ha='left', va='center', color = 'grey')

        ax.spines[['top','right']].set_visible(False)

        fig.savefig(os.path.join(fig_dir, f'utility_figure{eta}.png'), dpi=600, bbox_inches='tight')
