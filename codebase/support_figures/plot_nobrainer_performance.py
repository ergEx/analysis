
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from ..experiment_specs import condition_specs, sub_specs
from ..readingdata import reading_participant_passive_data


def plot_nobrainers(config, fig_size):
    """Plot no brainer performance for all participants.

    Parameters
    ----------
    config : The config file specifiying where data is located.

    Returns
    -------
        fig, axes, i.e. the figure parameters.

    """
    data_dir = config["data directory"]
    data_type = config["data_type"]
    data_variant = config["data_variant"]

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    try:
        input_path = config['input_path']
    except:
        input_path = data_dir

    try:
        extension = config['extension']
    except:
        extension = 'beh.csv'


    CONDITION_SPECS = condition_specs()
    SUBJECT_SPECS = sub_specs(data_type, data_variant, input_path,
                              exclusion_crit=[1], ignore_nobs=True)
    no_brainer_df = pd.DataFrame()


    for c, condition in enumerate(CONDITION_SPECS["condition"]):
        for i, subject in enumerate(SUBJECT_SPECS["id"]):
            _, no_brainer_participant_df = reading_participant_passive_data(
                    data_folder=input_path,
                    subject=subject,
                    first_run=SUBJECT_SPECS["first_run"][i][c],
                    bids_text=CONDITION_SPECS["bids_text"][c],
                    n_passive_runs=3,
                    input_path=input_path,
                    extension=extension
            )

            no_brainer_df = pd.concat([no_brainer_df, no_brainer_participant_df])


    grouped_nobs = no_brainer_df[['participant_id',
                                  'run', 'eta',
                                  'response_correct']].groupby(['participant_id',
                                                                'run', 'eta']).mean().reset_index()

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(fig_size[0] * 2, fig_size[1]))
    axes = axes.flatten()
    x_jitter = np.random.randn(np.unique(grouped_nobs.participant_id).shape[0]) * 0.05


    for et, ax in zip([0.0, 1.0], axes):

        for n, sub in enumerate(np.unique(grouped_nobs.participant_id)):

            tmp = grouped_nobs.query('participant_id == @sub and eta == @et')
            if tmp.query('run == 3').response_correct.values < 0.8:
                c = [1, 0, 0, 0.5]
            else:
                c = [0, 0, 1, 0.5]

            ax.plot(tmp.run + x_jitter[n], tmp.response_correct,
                    linestyle='dotted', marker='o', color=c, linewidth=0.5,
                    markersize=4)

        ax.axhline(0.8, linestyle='-', color=[0.5, 0.5, 0.5, 1.0])
        ax.axvline(1.5, linestyle='--', color=[0.25, 0.25, 0.25, 0.25])
        ax.axvline(2.5, linestyle='--', color=[0.25, 0.25, 0.25, 0.25])
        ax.set(xticks=[1, 2, 3], xticklabels=['1', '2', '3'])

    axes[0].set(title='Additive condition', ylabel='proportion correct', xlabel='run')
    axes[1].set(title='Multiplicative Condition', xlabel='run')

    return fig, axes
