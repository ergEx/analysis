import sys
import yaml
import os
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .utils import get_config_filename


def plot_sequential_bf(data, scale='medium', target='bf10'):
    import matplotlib.ticker as tck
    part = data['nsubs'].max()

    sub_data = data.query(f'scale == @scale and nsubs == @part').copy()

    sns.set_context('paper', font_scale=1.0)

    fig, ax = plt.subplots(figsize=(5.5,5.5))

    ax2 = plt.subplot2grid((8, 8), (0, 0), colspan=2, rowspan=2)
    ax2.pie(sub_data[['bf10', 'bf01']].values.ravel(), explode=[0.1, 0], labels=['$\mathrm{data}|\mathrm{H}_{1}$',
                                                                            '$\mathrm{data}|\mathrm{H}_{0}$'],
            wedgeprops = {'linewidth': 1, "edgecolor": 'white'}, startangle=0)

    BF10_text = '{:.2E}'.format(sub_data["bf10"].values[0])
    BF01_text = '{:.2E}'.format(sub_data["bf01"].values[0])

    ax3 = plt.subplot2grid((8, 8), (0, 3), colspan=4, rowspan=2)
    ax3.text(0, 0.75, '$\mathrm{BF}_{10} =$' + BF10_text, fontdict={'size':14})
    ax3.text(0, 0.25, '$\mathrm{BF}_{01} =$' + BF01_text, fontdict={'size':14})
    ax3.axis('off')

    ax = plt.subplot2grid((8, 8), (2, 0), rowspan=6, colspan=8)

    scatter_order = ['ultrawide', 'wide', 'abc', 'medium']

    data = data.sort_values('scale', key=np.vectorize(scatter_order.index))

    sns.scatterplot(data=data, x='nsubs', y=target, hue='scale', style='scale', ax=ax, s=100,
                   hue_order=['medium', 'wide', 'ultrawide'])

    min_lims = np.min(data[[target]].values.ravel())
    max_lims = np.max(data[[target]].values.ravel())

    min_lims = np.min([10 ** np.floor(np.log10(min_lims)), 10 ** -1])
    max_lims = np.max([10, 10 ** (np.floor(np.log10(max_lims)) + 1)])

    ax.set(ylabel='$\mathrm{BF}_{10}$', xlabel='n', yscale='log', ylim=[min_lims, max_lims])

    yticks = ax.get_yticks()

    ax.axhline(1, linewidth=1, linestyle='--', color='gray')
    ax.axhline(10, linewidth=1, linestyle='--', color='blue', alpha=0.5)

    yticks = 10 ** np.arange(np.log10(min_lims), np.log10(max_lims) + 1)
    yticklabels = [f'{int(i)}' for i in np.round(np.log10(yticks))]
    yticklabels = ['$10^{' + i + '}$' for i in yticklabels]

    yticklabels = [i if ((np.mod(n, 2) == 0) or (j in [0, 1])) else ''
                   for n, (i, j) in enumerate(zip(yticklabels,
                                                  np.log10(yticks).astype(int)))]

    ax.set(yticks=yticks, yticklabels=yticklabels)

    ax.set(xticks=np.arange(1, sub_data['nsubs'].values + 1, 2))
    ax.scatter(1 ,1, s=50, color='black')

    ax.legend(title='Prior width', loc='upper left')

    return fig, ax

def main(config_file, fig_dir=None):

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["bayesfactor_analysis"]["run"]:
        return

    print(f"\nRunning BayesFactor analysis")

    data_dir = config["data directory"]

    if fig_dir is None:
        fig_dir = config["figure directory"]

    target = config['bayesfactor_analysis']['target']

    subprocess.call(f'Rscript r_analyses/bayesian_t_test.R --path {data_dir}/ --mode {target}', shell=True)

    if config["bayesfactor_analysis"]["plot"]:

        # Q1 Sequential:
        q1_sequential = pd.read_csv(os.path.join(data_dir, 'q1_sequential_' + target + '.csv'), sep='\t')
        fig, ax = plot_sequential_bf(q1_sequential)
        fig.savefig(os.path.join(fig_dir, 'q1_sequential_' + target + '.pdf'),
                    bbox_inches='tight', dpi=600)

        # Q2 Sequential:
        q2_sequential = pd.read_csv(os.path.join(data_dir, 'q2_sequential_' + target + '.csv'), sep='\t')
        fig, ax = plot_sequential_bf(q2_sequential)
        fig.savefig(os.path.join(fig_dir, 'q2_sequential_' + target + '.pdf'),
                    bbox_inches='tight', dpi=600)