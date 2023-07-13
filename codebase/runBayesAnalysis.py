import sys
import yaml
import os
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .utils import get_config_filename


def plot_sequential_bf(data, scale='medium', part=11):

    sub_data = data.query(f'scale == @scale and nsubs == @part').copy()

    sns.set_context('paper', font_scale=1.0)

    fig, ax = plt.subplots(figsize=(5.5,5.5))

    ax = plt.subplot2grid((8, 8), (2, 0), rowspan=6, colspan=8)

    sns.scatterplot(data=data, x='nsubs', y='bf10', hue='scale', style='scale', ax=ax, s=100)

    ax.axhline(1, linewidth=1, linestyle='--', color='gray')
    ax.axhline(10, linewidth=1, linestyle='--', color='blue', alpha=0.5)

    min_lims = np.min(data[['bf10']].values.ravel())
    max_lims = np.max(data[['bf10']].values.ravel())

    min_lims = np.min([min_lims, 10 ** -1])
    max_lims = 10 ** (np.floor(np.log10(max_lims)) + 1)
    ax.set(ylim=[min_lims, max_lims], ylabel='$\mathrm{BF}_{10}$', xlabel='n')
    ax.set_yscale('log')

    ax.legend(title='Prior width', loc='upper left')

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

    return fig, ax


def main(config_file):

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["bayesfactor_analysis"]["run"]:
        return

    print(f"\nRunning BayesFactor analysis")

    data_dir = config["data directory"]
    fig_dir = config["figure directory"]
    target = config['bayesfactor_analysis']['target']

    subprocess.call(f'rscript r_analyses/bayesian_t_test.R --path {data_dir}/ --mode {target}', shell=True)

    if config["bayesfactor_analysis"]["plot"]:

        # Q1 Sequential:
        q1_sequential = pd.read_csv(os.path.join(data_dir, 'q1_sequential_' + target + '.csv'), sep='\t')
        q1_sequential = q1_sequential.query('scale == "medium"').copy()
        fig, ax = plot_sequential_bf(q1_sequential)
        fig.savefig(os.path.join(fig_dir, 'q1_sequential_' + target + '.pdf'),
                    bbox_inches='tight', dpi=600)

        # Q2 Sequential:
        q2_sequential = pd.read_csv(os.path.join(data_dir, 'q2_sequential_' + target + '.csv'), sep='\t')
        fig, ax = plot_sequential_bf(q2_sequential)
        fig.savefig(os.path.join(fig_dir, 'q2_sequential_' + target + '.pdf'),
                    bbox_inches='tight', dpi=600)

if __name__ == "__main__":
    config_file = get_config_filename(sys.argv)
    main(config_file)
