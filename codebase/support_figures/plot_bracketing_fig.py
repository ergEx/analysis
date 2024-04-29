import os

import matplotlib.pyplot as plt
import numpy as np


def bracketing_fig(fig_dir):

    cm = 1/2.54
    fig_size = (7.5 * cm , 6 * cm)

    #fig 1
    fig1, ax1 = plt.subplots()

    ax1.spines[['left', 'right']].set_visible(False)
    ax1.spines[['top']].set_visible(False)
    ax1.set_yticks([])
    ax1.set_xlim(-3, 4)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_xlabel('$\eta^{ind}}$')

    x1, y1 = -1, 0.5
    x2, y2 = 3, 0.5

    arrow1 = ax1.arrow(x1, y1, 0.3, y2-y1, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
    arrow2 = ax1.arrow(x2, y2, -0.3, y1-y2, head_width=0.05, head_length=0.1, fc='orange', ec='orange')

    fig1.set_size_inches(fig_size)
    fig1.savefig(os.path.join(fig_dir, f"bracketing_1.pdf"), dpi=600, bbox_inches='tight')

    #fig 2 and 3
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    ax2.spines[['left', 'right']].set_visible(False)
    ax2.spines[['top']].set_visible(False)
    ax2.set_yticks([])
    ax2.set_xlim(-3, 4)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_xlabel('$\eta^{ind}}$')

    ax3.spines[['left', 'right']].set_visible(False)
    ax3.spines[['top']].set_visible(False)
    ax3.set_yticks([])
    ax3.set_xlim(-3, 4)
    ax3.set_ylim(-0.2, 1.2)
    ax3.set_xlabel('$\eta^{ind}}$')

    for _ in range(30):
        x1, y1 = np.random.uniform(-3, 0.5, 1), 0.0
        x2, y2 = np.random.uniform(0.5, 4, 1), 1.0
        arrow1 = ax2.arrow(x1-0.3, y1, 0.3, 0, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        arrow2 = ax2.arrow(x2+0.3, y2, -0.3, 0, head_width=0.05, head_length=0.1, fc='orange', ec='orange')
        arrow1 = ax3.arrow(x1-0.3, y1, 0.3, 0, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        arrow2 = ax3.arrow(x2+0.3, y2, -0.3, 0, head_width=0.05, head_length=0.1, fc='orange', ec='orange')

    x = np.linspace(-3, 4, 100)
    y = np.repeat(np.arange(2), len(x) // 2)
    ax2.step(x, y, where='post')

    ax2.axvline(0.5, ymax=0.5, linestyle='--', color='k')
    ax2.axhline(0.5, linestyle='--', color='k')

    fig2.set_size_inches(fig_size)
    fig2.savefig(os.path.join(fig_dir, f"bracketing_2.pdf"), dpi=600, bbox_inches='tight')

    #fig 3 (extra)
    for _ in range(15):
        x1, y1 = abs(np.random.normal(0.5, 1, 1)), 0.0
        x2, y2 = -abs(np.random.normal(0.5, 1, 1)), 1.0
        arrow1 = ax3.arrow(x1-0.3, y1, 0.3, 0, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        arrow2 = ax3.arrow(x2+0.3, y2, -0.3, 0, head_width=0.05, head_length=0.1, fc='orange', ec='orange')

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x = np.linspace(-3, 4, 100)
    y = sigmoid(x*5-2.4)
    ax3.plot(x, y)

    ax3.axhline(0.5, linestyle='--', color='k')
    ax3.axvline(0.5, ymax=0.5, linestyle='--', color='k')

    fig3.set_size_inches(fig_size)
    fig3.savefig(os.path.join(fig_dir, f"bracketing_3.pdf"), dpi=600, bbox_inches='tight')
