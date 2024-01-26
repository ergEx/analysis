import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
from scipy.stats import gaussian_kde

sns.set_context('paper', font_scale=1.1) #, rc=rcParamsDefault)
cm = 1/2.54  # centimeters in inches (for plot size conversion)
fig_size = (6.5 * cm , 5.75 * cm)

plt.rcParams.update({
    "text.usetex": True})


def posterior_dist_plot(fig, ax, data_no_pooling, data_pooling, colors, colors_alpha,
                        n_conditions, n_agents, labels, LIMITS, x_label):
    ax2 = ax.twinx()
    maxi = np.zeros([n_conditions,n_agents,2])
    for c in range(n_conditions):
        for i in range(n_agents):
            data_tmp = data_no_pooling[:,:,i,c].flatten()
            sns.kdeplot(data_tmp, ax = ax, color = colors_alpha[c], linewidth=0.5)
            kde = gaussian_kde(data_tmp)

            maxi[c,i,0] = data_tmp[np.argmax(kde.pdf(data_tmp))]
            maxi[c,i,1] = kde.pdf(maxi[c,i,0])

        sns.kdeplot(data_pooling[:,:,c].ravel(), ax = ax2,
                    color = colors[c], linestyle = '-', label = labels[c],
                    linewidth=1.5)

    ax.set(xlim = LIMITS, xlabel = x_label, ylabel = '')
    ax.tick_params(axis='y', which='both', left=False, right=False,
                   labelleft=False, labelright=False)
    ax.spines[['left', 'top','right']].set_visible(False)

    ax2.set(ylabel = '')
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    ax2.spines[['left', 'top', 'right']].set_visible(False)
    ax2.legend(loc='upper left', fontsize=10)
    return fig, ax, ax2, maxi


def posterior_dist_2dplot(fig, ax, data_no_pooling, colors_alpha, LIMITS, maxi):

    sns.kdeplot(x=data_no_pooling[:,:,:,0].ravel(),
                y=data_no_pooling[:,:,:,1].ravel(),
                cmap="coolwarm", fill=True, ax = ax)

    sns.lineplot(x=LIMITS, y=LIMITS, color='black', linestyle='--', ax=ax, alpha = 0.3)
    ax.axvline(0, color=colors_alpha[0], linestyle='--')
    ax.axhline(1, color=colors_alpha[1], linestyle='--')
    ax.set(xlim = LIMITS, ylim = LIMITS, xlabel = r"$\eta^{\mathrm{add}}$", ylabel = r"$\eta^{\mathrm{mul}}$")
    ax.spines[['top','right']].set_visible(False)

    if maxi is not None:
        ax.scatter(x=maxi[0, :, 0], y=maxi[1, :, 0], marker='x',
                   color='black', alpha = 0.3, label = 'MAP estimates')
        ax.legend(loc='lower right')
    return fig, ax


def jasp_like_raincloud(data, col_name1, col_name2,
                        palette=[[0, 0, 1, 0.25], [1, 0, 0, 0.25]],
                        alpha=0.5, colors=None, fig_size=fig_size):
    """Recreates raincloud plots, similarly to the ones in JASP

    Args:
        data (pd.DataFrame): Jasp input file
        col_name1 (str): Column name 1, assumed to be additive condition.
        col_name2 (str): Column name 2, assumed to be multiplicative condition.
        palette (list, optional): Color palette for plots. Defaults to ['blue', 'red'].
        ylimits (list, optional): Limits of the yaxis. Defaults to [-0.1, 1.2].

    Returns:
        fig, axes: figure and axes of the raincloud plots
    """

    fig = plt.figure(figsize=fig_size) #, figsize=fig_size)

    ax1 = plt.subplot2grid(shape=(6, 3), loc=(0, 0), rowspan=4, colspan=2)
    ax2 = plt.subplot2grid(shape=(6, 3), loc=(0, 2), rowspan=4, colspan=1)
    ax3 = plt.subplot2grid(shape=(6, 3), loc=(4, 0), rowspan=2, colspan=3)

    axes = [ax1, ax2, ax3]


    sub_data = data[[col_name1, col_name2]].copy()
    sub_data = sub_data.melt(value_vars=[col_name1, col_name2],
                             var_name='Condition', value_name='Estimate')
    sub_data['x'] = 1

    d1 = data[[col_name1]].values
    d2 = data[[col_name2]].values

    x_jitter = np.random.randn(*d1.shape) * 0.05
    xj_mean = x_jitter.mean()

    for n, (i, j) in enumerate(zip(d1, d2)):
        if colors is not None:
            axes[0].plot([1 + x_jitter[n], 2 + x_jitter[n]], [i, j], color=colors[n],
                         linewidth=0.2)
        else:
            axes[0].plot([1 + x_jitter[n], 2 + x_jitter[n]], [i, j], color=[0.1, 0.1, 0.1, 0.25],
                         linewidth=0.2)

    if colors is not None:
        axes[0].scatter(np.ones(d1.shape) + x_jitter, d1, color=colors)
        axes[0].scatter(np.ones(d1.shape) + 1 + x_jitter, d2, color=colors)
    else:
        axes[0].scatter(np.ones(d1.shape) + x_jitter, d1, color=palette[0])
        axes[0].scatter(np.ones(d1.shape) + 1 + x_jitter, d2, color=palette[1])
    # ylim=ylimits,
    axes[0].set(xticks=[1 + xj_mean, 2 + xj_mean],
                xticklabels=['Additive\ncondition', 'Multiplicative\ncondition'],
                ylabel='Risk aversion parameter')
    axes[0].spines[['right', 'top']].set_visible(False)

    pt.half_violinplot(x='x', y='Estimate', hue='Condition', data=sub_data, ax=axes[1],
                 palette=palette, alpha=alpha)

    axes[1].get_legend().remove()
    axes[1].set(ylabel='', xlabel='', xticklabels=[], xticks=[], yticks=[])
    # axes[1].invert_xaxis()
    axes[1].spines[['right', 'top', 'left', 'bottom']].set_visible(False)

    for artist in axes[1].patches:
        artist.set_alpha(alpha)

    diff = d1 - d2
    pt.RainCloud(y=diff, x=None, axes=axes[2], orient='h')
    axes[2].spines[['right', 'top']].set_visible(False)
    axes[2].set(xlabel='$\Delta$ Risk aversion parameter')
    axes[2].set_ylim([0.2, -0.7125])
    axes[2].set_xlim([np.min(diff) - 0.2, np.max(diff) + 0.2])
    axes[2].axvline(x=0, color=[0.25, 0.25, 0.25, 0.25], linestyle='--')
    plt.tight_layout()

    return fig, axes


def jasp_like_correlation(data, col_name1, col_name2, lim_offset=0.01, colors=None):
    """Correlation plot.

    Args:
        data (pd.DataFrame): Jasp input file
        col_name1 (str): Column name 1, assumed to be additive condition (x-axis).
        col_name2 (str): Column name 2, assumed to be multiplicative condition (y-axis).
        lim_offset (float, optional): Additional space to y and x-axis. Defaults to 0.01.

    Returns:
        fig, ax: Figure and axes objects.
    """

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    if colors is not None:
        ax.scatter(x=data[col_name1], y=data[col_name2], c=colors)
        plot_dots = False
    else:
        plot_dots = True

    sns.regplot(x=col_name1, y=col_name2, data=data, ax=ax, scatter=plot_dots)

    ax.set(ylabel='$\hat{\eta}^{\mathrm{mul}}$', xlabel='$\hat{\eta}^{\mathrm{add}}$')

    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())
    lim_offset = np.array([lim_offset * -1, lim_offset])

    ax.set(xlim = xlim + lim_offset, ylim=ylim + lim_offset)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.spines[['right', 'top']].set_visible(False)

    return fig, ax

def model_select_plot(z, models, data_dir, name):

    n_chains, n_samples, n_participants = z.shape

    z_mod = np.mod(z, len(models))
    z_mod[z_mod == 0] = len(models)

    z_i_mod = z_mod.reshape((n_chains * n_samples, n_participants))

    counts = np.zeros([len(models), n_participants])
    bin_edges = np.arange(1, len(models) + 2)

    for col in range(n_participants):
        counts[:, col], _ = np.histogram(z_i_mod[:, col], bins=bin_edges)

    counts += 1  # Add 1 to all counts to avoid division by zero

    proportions = counts / np.sum(counts, axis=0)

    np.savetxt(os.path.join(data_dir,f'proportions_{name}.txt'), np.log(proportions))

    df = pd.DataFrame(proportions.T, columns=models)

    fig, ax = plt.subplots(1, 1, figsize = (15,5))
    sns.heatmap(df, cmap='gray_r', yticklabels=False, cbar=False, ax=ax)
    ax.set_ylabel('Participants')

    fig2, ax2 = plt.subplots(1, 1, figsize = (15,5))
    D = dict(sorted(Counter(z_i_mod.ravel()).items()))
    ax2.bar(range(len(D)), list(D.values()), align='center')
    ax2.set_xticks(range(len(D)), models)
    ax2.set(yticks = [])
    ax2.spines[['left', 'top', 'right']].set_visible(False)

    return fig, ax, fig2, ax2

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