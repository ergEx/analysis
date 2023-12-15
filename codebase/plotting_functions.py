from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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
            sns.kdeplot(data_tmp, ax = ax, color = colors_alpha[c])
            kde = gaussian_kde(data_tmp)

            maxi[c,i,0] = data_tmp[np.argmax(kde.pdf(data_tmp))]
            maxi[c,i,1] = kde.pdf(maxi[c,i,0])

        sns.kdeplot(data_pooling[:,:,c].ravel(), ax = ax2,
                    color = colors[c], linestyle = '-', label = labels[c])

    ax.set(xlim = LIMITS, xlabel = x_label, ylabel = '')
    ax.tick_params(axis='y', which='both', left=False, right=False,
                   labelleft=False, labelright=False)
    ax.spines[['left', 'top','right']].set_visible(False)

    ax2.set(ylabel = '')
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    ax2.spines[['left', 'top', 'right']].set_visible(False)
    ax2.legend(loc='upper right')
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


def jasp_like_raincloud(data, col_name1, col_name2, palette=['blue', 'red'],
                        ylimits=[-0.1, 1.2], alpha=0.5, colors=None):
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

    fig, axes = plt.subplots(1, 2, sharey=False, figsize=fig_size)
    axes = axes.flatten()

    sub_data = data[[col_name1, col_name2]].copy()
    sub_data = sub_data.melt(value_vars=[col_name1, col_name2], var_name='Condition', value_name='Estimate')
    sub_data['x'] = 1

    d1 = data[[col_name1]].values
    d2 = data[[col_name2]].values

    x_jitter = np.random.rand(*d1.shape) * 0.1
    xj_mean = x_jitter.mean()

    for n, (i, j) in enumerate(zip(d1, d2)):
        if colors is not None:
            axes[0].plot([1 + x_jitter[n], 2 + x_jitter[n]], [i, j], color=colors[n])
        else:
            axes[0].plot([1 + x_jitter[n], 2 + x_jitter[n]], [i, j], color=[0.1, 0.1, 0.1, 0.25])

    if colors is not None:
        axes[0].scatter(np.ones(d1.shape) + x_jitter, d1, color=colors)
        axes[0].scatter(np.ones(d1.shape) + 1 + x_jitter, d2, color=colors)
    else:
        axes[0].scatter(np.ones(d1.shape) + x_jitter, d1, color=palette[0])
        axes[0].scatter(np.ones(d1.shape) + 1 + x_jitter, d2, color=palette[1])

    axes[0].set(ylim=ylimits, xticks=[1 + xj_mean, 2 + xj_mean],
                xticklabels=['Additive\ncondition', 'Multiplicative\ncondition'],
                ylabel='Risk aversion parameter')
    axes[0].spines[['right', 'top']].set_visible(False)

    pt.RainCloud(x='x', y='Estimate', hue='Condition', data=sub_data, ax=axes[1],
                 palette=palette, alpha=alpha)

    axes[1].get_legend().remove()
    axes[1].set(ylim=ylimits, ylabel='', xlabel='', xticklabels=[], xticks=[], yticks=[])
    axes[1].invert_xaxis()
    axes[1].spines[['right', 'top', 'left', 'bottom']].set_visible(False)

    for artist in axes[1].patches:
            artist.set_alpha(alpha)

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


def paired_swarm_plot(data, col_name1, col_name2, palette=['blue', 'red'],
                        ylimits=[-0.1, 1.2], alpha=0.5, colors=None):
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

    fig, axes = plt.subplots(1, 1, sharey=False, figsize=(fig_size[1], fig_size[1]))

    sub_data = data[[col_name1, col_name2]].copy()
    sub_data = sub_data.melt(value_vars=[col_name1, col_name2], var_name='Condition', value_name='Estimate')
    sub_data['x'] = 1

    d1 = data[[col_name1]].values
    d2 = data[[col_name2]].values

    x_jitter = np.random.rand(*d1.shape) * 0.1
    xj_mean = x_jitter.mean()

    for n, (i, j) in enumerate(zip(d1, d2)):
        #if colors is not None:
        #    axes[0].plot([1 + x_jitter[n], 2 + x_jitter[n]], [i, j], color=colors[n])
        #else:
        axes.plot([1 + x_jitter[n], 2 + x_jitter[n]], [i, j], color=[0.1, 0.1, 0.1, 0.25], linewidth=0.5)

    if colors is not None:
        axes.scatter(np.ones(d1.shape) + x_jitter, d1, color=colors)
        axes.scatter(np.ones(d1.shape) + 1 + x_jitter, d2, color=colors)
    else:
        axes.scatter(np.ones(d1.shape) + x_jitter, d1, color=palette[0])
        axes.scatter(np.ones(d1.shape) + 1 + x_jitter, d2, color=palette[1])

    axes.set(ylim=ylimits, xticks=[1 + xj_mean, 2 + xj_mean],
                xticklabels=['$\eta^{\mathrm{add}}$', '$\eta^{\mathrm{mul}}$'],
                ylabel='Risk aversion parameter')
    axes.spines[['right', 'top']].set_visible(False)

    return fig, axes


def plot_individual_heatmaps(data, colors, hue, limits = [-3,3],
                             x_fiducial=[], y_fiducial=[]):

    h1 = sns.jointplot(
        data=data,
        x=data[:,0],
        y=data[:,1],
        hue=hue,
        kind="kde",
        alpha=0.7,
        fill=True,
        palette = sns.color_palette(colors),
        xlim = limits,
        ylim = limits,
        legend = False
        )

    h1.set_axis_labels("$\eta^{\mathrm{add}}$", "$\eta^{\mathrm{mul}}$")
    ticks = np.arange(limits[0], limits[1] + 0.5, 0.5)
    h1.ax_joint.set_xticks(ticks)
    h1.ax_joint.set_yticks(ticks)

    if len(ticks) > 5:
        ticklabels = [f'{ii}' if ii == np.round(ii) else '' for ii in ticks ]
    else:
        ticklabels = ticks

    h1.ax_joint.set(xticklabels=ticklabels, yticklabels=ticklabels)

    h2 = sns.lineplot(x=limits, y=limits, color=[0.25, 0.25, 0.25, 0.25], linestyle='--', ax=h1.ax_joint)

    for xl in x_fiducial:
        fid_color = 'blue'
        h1.ax_joint.axvline(xl, color=fid_color, alpha=0.5, linestyle='--')

    for yl in y_fiducial:
        fid_color = 'red'
        h1.ax_joint.axhline(yl, color=fid_color, alpha=0.5, linestyle='--')

    h1.fig.set_size_inches(fig_size[1], fig_size[1])
    return h1


def plot_single_kde(data, ax, limits = [-3, 3], colors = ['blue', 'red'], labels = ['Additive', 'Multiplicative'], x_fiducials=[]):
    maxi = np.empty([2,2])
    for i in range(2):
        sns.kdeplot(data[i], color=colors[i], label=labels[i], fill=True, ax=ax)

        kde = gaussian_kde(data[i])

        maxi[i,0] = data[i][np.argmax(kde.pdf(data[i]))]
        maxi[i,1] = kde.pdf(maxi[i,0])

    ax.axvline(maxi[0,0], ymax=maxi[0,1] / (ax.get_ylim()[1]), color='black', linestyle='--')
    ax.axvline(maxi[1,0], ymax=maxi[1,1] / (ax.get_ylim()[1]), color='black', linestyle='--')
    ax.plot([], ls="--", color="black", label="Estimates")
    ax.legend(loc="upper left", fontsize=6)
    ticks = np.arange(limits[0], limits[1] + 0.5, 0.5)

    if len(ticks) > 5:
        ticklabels = [f'{ii}' if ii == np.round(ii) else '' for ii in ticks ]
    else:
        ticklabels = ticks

    ax.set(
        title="",
        xlabel="$\eta$",
        ylabel="",
        xlim=limits,
        yticks=[],
        xticks=ticks,
        xticklabels=ticklabels)


    for xl in x_fiducials:
        # This is a bit convoluted, but just in case we want to use different colors for the fiducials.
        if xl == 0:
            fid_color = colors[xl]
        elif xl == 1:
            fid_color = colors[xl]

        ax.axvline(xl, color=fid_color, linestyle='--', alpha=0.5)

    ax.spines[['right', 'top']].set_visible(False)


    return ax
