import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from .utils import posterior_dist_2dplot, read_Bayesian_output


def create_grid_sim_plot(config_file):
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    plt.rcParams.update({
        "text.usetex": True})
    sns.set_context('paper', font_scale=1.1)

    cm = 1/2.54  # centimeters in inches (for plot size conversion)
    fig_size = (6.5 * cm , 5.75 * cm)

    LIMITS = [-1,2]

    data_dir = "data/0_simulation/grid/"
    fig_dir = "figs/0_simulation/grid/"

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    quality_dictionary = {'chains': [2,4,4,4], 'samples': [5e1,5e2,5e3,1e4,2e4], 'manual_burnin': [1e1,1e3,1e4,2e4,4e4]}
    n_agents = config["n_agents"]
    burn_in = int(quality_dictionary['manual_burnin'][config['qual'] - 1])
    n_samples = int(quality_dictionary['samples'][config['qual'] - 1])
    n_chains = int(quality_dictionary['chains'][config['qual'] - 1])
    n_conditions = config["n_conditions"]

    colors_alpha = [np.array([0, 0, 1, 0.3], dtype=float), np.array([1, 0, 0, 0.3], dtype=float)]

    types = ['eta_n05', 'eta_00', 'eta_05', 'eta_10', 'eta_15', 'time_optimal']

    fig_bracketing, ax_bracketing = plt.subplots(1, 1, figsize=fig_size)
    fig_bayesian_no_pooling, ax_bayesian_no_pooling = plt.subplots(1, 1, figsize=fig_size)
    fig_bayesian_partial_pooling, ax_bayesian_partial_pooling = plt.subplots(1, 1, figsize=fig_size)

    for i, type in enumerate(types):
        data_dir_tmp = data_dir + type

        #Bracketing
        bracketing_overview = pd.read_csv(os.path.join(data_dir_tmp, "bracketing_overview.csv"), sep = '\t')
        df_no_pooling = bracketing_overview[bracketing_overview.participant != 'all']
        eta_i = np.zeros(n_chains, n_samples, n_agents, n_conditions)
        for ch in range(n_chains):
            for c, con in enumerate(df_no_pooling['dynamic'].unique()):
                for i, participant in enumerate(df_no_pooling['participant'].unique()):
                    tmp_df_i = df_no_pooling.query('participant == @participant and dynamic == @con')
                    eta_i[ch,:,i,c] = np.random.normal(tmp_df_i.log_reg_decision_boundary, tmp_df_i.log_reg_std_dev, n_samples-burn_in)

        fig_bracketing, ax_bracketing = posterior_dist_2dplot(fig_bracketing, ax_bracketing, eta_i, colors_alpha, LIMITS, None)

        #Bayesian
        bayesian_samples_no_pooling = read_Bayesian_output(
                        os.path.join(data_dir_tmp, "Bayesian_JAGS_parameter_estimation_no_pooling.mat")
                        )
        eta_i = bayesian_samples_no_pooling["eta_i"]

        fig_bayesian_no_pooling, ax_bayesian_no_pooling = posterior_dist_2dplot(fig_bayesian_no_pooling, ax_bayesian_no_pooling, eta_i, colors_alpha, LIMITS, None)

        bayesian_samples_no_pooling = read_Bayesian_output(
                        os.path.join(data_dir_tmp, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                        )
        eta_i = bayesian_samples_no_pooling["eta_i"]

        fig_bayesian_partial_pooling, ax_bayesian_partial_pooling = posterior_dist_2dplot(fig_bayesian_partial_pooling, ax_bayesian_partial_pooling, eta_i, colors_alpha, LIMITS, None)

    fig_bracketing.savefig(os.path.join(fig_dir, 'simulations_bracketing.png'), dpi=600, bbox_inches='tight')
    fig_bayesian_no_pooling.savefig(os.path.join(fig_dir, 'simulations_bayesian_no_pooling.png'), dpi=600, bbox_inches='tight')
    fig_bayesian_partial_pooling.savefig(os.path.join(fig_dir, 'simulations_bauesian_partial_pooling.png'), dpi=600, bbox_inches='tight')
