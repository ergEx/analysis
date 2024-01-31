import os

import matplotlib.pyplot as plt
import numpy as np

from ..utils import isoelastic_utility

plt.rcParams.update({
    "text.usetex": True})

cm = 1/2.54  # centimeters in inches (for plot size conversion)
fig_size = (13 * cm , (5.75*2) * cm)


def draw_brace(ax, span, pos, text, col, orientation):
    """Draws an annotated brace on the axes."""
    ax_min, ax_max = ax.get_xlim() if orientation == 'horizontal' else ax.get_ylim()
    _span = ax_max - ax_min

    opp_min, opp_max = ax.get_ylim() if orientation == 'horizontal' else ax.get_xlim()
    opp_span = opp_max - opp_min

    beta_factor = 300.

    resolution = int((span[1]-span[0])/_span*100)*2+1
    beta = beta_factor/_span

    x = np.linspace(span[0], span[1], resolution)
    _half = x[:int(resolution/2)+1]
    opp_half_brace = (1/(1.+np.exp(-beta*(_half-_half[0])))
                    + 1/(1.+np.exp(-beta*(_half-_half[-1]))))
    y = np.concatenate((opp_half_brace, opp_half_brace[-2::-1]))
    y = pos + (.05*y - .01)*opp_span # adjust vertical position

    ax.autoscale(False)

    if orientation == 'horizontal':
        ax.plot(x, y, color=col, lw=1)
        ax.text((span[1]+span[0])/2., pos+.07*opp_span, text, ha='center', va='bottom', color = col)
    else:
        ax.plot(y, x, color=col, lw=1)
        ax.text(pos + .07*opp_span, (span[1] + span[0])/2., text, ha='left', va='center', color = col)


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
