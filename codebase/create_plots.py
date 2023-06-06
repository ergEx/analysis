#%% # -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from .utils import plot_individual_heatmaps, plot_single_kde, read_Bayesian_output


def main(config_file):
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["plots"]["run"]:
        return

    print(f"\nPLOTTING")

    data_dir = config["data directoty"]
    fig_dir = config["figure directoty"]
    data_type = config["data_type"]

    n_agents = config["n_agents"]
    n_samples = config["n_samples"]
    n_chains = config["n_chains"]
    n_conditions = config["n_conditions"]

    stages = config["plots"]

    title_dict={0: "Additive", 1: "Multiplicative"}
    soft_limits = {0.0: [-500, 2_500], 1.0: [64 , 15_589]}

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in np.linspace(0, 1, n_agents)]

    if stages['plot_passive']:
        if data_type != 'real_data':
            print('There is no passive trajectories for simulated data')
        else:
            df_passive = pd.read_csv(os.path.join(data_dir, "all_passive_phase_data.csv"), sep="\t")
            fig, ax = plt.subplots(2,1)
            for c, con in enumerate(set(df_passive.eta)):
                tmp_df = df_passive.query("eta == @con")
                pivoted_df = tmp_df.pivot(index='trial', columns='participant_id', values='wealth')
                for i, participant in enumerate(pivoted_df.columns):
                    ax[c].plot(pivoted_df.index, pivoted_df[participant],  color = colors[i])
                ax[c].set(title = title_dict[c],xlabel="Trial", ylabel="Wealth")
                if c == 1:
                    ax[c].set(yscale="log", ylabel="log Wealth (log)")

                for i in [45, 45 * 2]:
                    ax[c].axvline(x=i, linestyle="--", color="grey")
                ax[c].plot([], linestyle="--", color="grey", label='reset')
                ax[c].legend(loc="upper left")

            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, '01_passive_trajectories.png'))

    if stages['plot_no_brainers']:
        if data_type != "real_data":
            print('There is no no-brainer data for simulated data')
        else:
            df_no_brainer = pd.read_csv(os.path.join(data_dir, "all_no_brainer_data.csv"), sep="\t")
            fig, ax = plt.subplots(1,2)
            for c, con in enumerate(set(df_no_brainer.eta)):
                df_rankings_copy = df_no_brainer.copy()
                df_rankings_copy["trial_bins"] = pd.cut(
                    df_rankings_copy["trial"], bins=[40, 70, 110, 160], labels=["First", "Second", "Third"]
                )
                df_prop = (
                    df_rankings_copy.groupby(["participant_id", "trial_bins"])
                    .mean()["response_correct"]
                    .reset_index()
                )
                sns.scatterplot(
                    x="trial_bins",
                    y="response_correct",
                    hue="participant_id",
                    palette=colors,
                    data=df_prop,
                    s=20,
                    marker="x",
                    ax=ax[c],
                )
                ax[c].set(
                    ylim=(0, 1), ylabel="Proportion of correct rankings", xlabel="", title=title_dict[c]
                )
                ax[c].legend().remove()
                ax[c].axhline(y=0.8, color="black", linestyle="--")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, '02_no_brainers.png'))

    if stages['plot_active']:
        if data_type != 'real_data':
            print('There is no passive trajectories for simulated data')
        else:
            df_active = pd.read_csv(os.path.join(data_dir, "all_active_phase_data.csv"), sep="\t")
            fig, ax = plt.subplots(2,1)
            for c, con in enumerate(set(df_active.eta)):
                tmp_df = df_active.query("eta == @con")
                pivoted_df = tmp_df.pivot(index='trial', columns='participant_id', values='wealth')
                for i, participant in enumerate(pivoted_df.columns):
                    ax[c].plot(pivoted_df.index, pivoted_df[participant], color = colors[i])
                ax[c].set(title=title_dict[c],xlabel="Trial", ylabel="Wealth")
                if c == 1:
                    ax[c].set(yscale="log", ylabel="log Wealth (log)")

                ax[c].axhline(soft_limits[c][1], linestyle="--", color="grey", label='upper limit')
                ax[c].axhline(soft_limits[c][0], linestyle="--", color="grey", label='lower limit')
                #ax[c].legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, '03_active_trajectories.png'))

    if stages['plot_riskaversion_bracketing']:
        #Full pooling
        bracketing_overview = pd.read_csv(os.path.join(data_dir, "bracketing_overview.csv"), sep = '\t')

        df_tmp = bracketing_overview[bracketing_overview.participant == 'all']
        add = np.random.normal(df_tmp[df_tmp.dynamic == 0.0].log_reg_decision_boundary, df_tmp[df_tmp.dynamic == 0.0].log_reg_std_dev, n_samples * n_conditions)
        mul = np.random.normal(df_tmp[df_tmp.dynamic == 1.0].log_reg_decision_boundary, df_tmp[df_tmp.dynamic == 1.0].log_reg_std_dev, n_samples * n_chains)
        fig, ax = plt.subplots(1, 1)
        ax = plot_single_kde([add,mul], ax)
        fig.savefig(os.path.join(fig_dir, '04_riskaversion_full_pooling_group_bracketing'))

        #No pooling
        df_tmp = bracketing_overview[bracketing_overview.participant != 'all']
        etas = np.empty([n_agents,5000*4,2])
        for i, participant in enumerate(list(set(df_tmp.participant))):
            for c, con in enumerate(list(set(df_tmp.dynamic))):
                tmp_df = df_tmp.query('participant == @participant and dynamic == @con')
                etas[i,:,c] = np.random.normal(tmp_df.log_reg_decision_boundary, tmp_df.log_reg_std_dev, 5000*4)
        etas_log_r = np.reshape(etas, (n_agents * n_samples * n_chains, n_conditions))
        h1 = plot_individual_heatmaps(etas_log_r, colors, hue = np.repeat(np.arange(n_agents), 4 * 5000))
        h1.savefig(os.path.join(fig_dir, '05_riskaversion_no_pooling_individual_bracketing'))

    if stages['plot_riskaversion_bayesian']:
        # full pooling
        # group
        bayesian_samples_full_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_full_pooling.mat")
                    )
        eta_group = bayesian_samples_full_pooling["eta"]
        fig, ax = plt.subplots(1, 1)
        ax = plot_single_kde([eta_group[:,:,0].flatten(),eta_group[:,:,1].flatten()], ax)
        fig.savefig(fig_dir,'06_riskaversion_full_pooling_group_bayesian.png')

        # partial pooling
        # group
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                )
        eta_group = bayesian_samples_partial_pooling["eta_g"]
        fig, ax = plt.subplots(1, 1)
        ax = plot_single_kde([eta_group[:,:,0].flatten(),eta_group[:,:,1].flatten()], ax)
        fig.savefig(fig_dir,'07_riskaversion_partial_pooling_group_bayesian.png')

        #individual
        eta_i = bayesian_samples_partial_pooling["eta"]
        eta_i_part_t = eta_i.transpose((2, 0, 1, 3))
        eta_i_part_t_r = np.reshape(eta_i_part_t, (n_agents * n_samples * n_chains, n_conditions))
        h1 = plot_individual_heatmaps(eta_i_part_t_r, colors)
        h1.savefig(os.path.join(fig_dir, f"08_riskaversion_partial_pooling_individual_bayesian.pdf"))

        # no pooling
        # individual
        bayesian_samples_no_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_no_pooling.mat")
                )
        eta_i = bayesian_samples_no_pooling["eta"]
        eta_i_t = eta_i.transpose((2, 0, 1, 3))
        eta_i_t_r = np.reshape(eta_i_t, (n_agents * n_samples * n_chains, n_conditions))
        h1 = plot_individual_heatmaps(eta_i_t_r, colors)
        h1.savefig(os.path.join(fig_dir, f"09_riskaversion_no_pooling_individual_bayesian.pdf"))

    if stages['plot_sensitivity_bayesian']:
        # full pooling
        # group
        bayesian_samples_full_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_full_pooling.mat")
                    )
        beta_group = bayesian_samples_full_pooling["beta"]
        fig, ax = plt.subplots(1, 1)
        ax = plot_single_kde([beta_group[:,:,0].flatten(),beta_group[:,:,1].flatten()], ax)
        fig.savefig(fig_dir,'10_sensitivity_full_pooling_group_bayesian.png')

        # partial pooling
        # group
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                )
        beta_group = bayesian_samples_partial_pooling["beta_g"]
        fig, ax = plt.subplots(1, 1)
        ax = plot_single_kde([beta_group[:,:,0].flatten(),beta_group[:,:,1].flatten()], ax)
        fig.savefig(fig_dir,'11_sensitivity_partial_pooling_group_bayesian.png')

        #individual
        beta_i = bayesian_samples_partial_pooling["beta"]
        beta_i_part_t = beta_i.transpose((2, 0, 1, 3))
        beta_i_part_t_r = np.reshape(beta_i_part_t, (n_agents * n_samples * n_chains, n_conditions))
        h1 = plot_individual_heatmaps(beta_i_part_t_r, colors)
        h1.savefig(os.path.join(fig_dir, f"12_sensitivity_partial_pooling_individual_bayesian.pdf"))

        # no pooling
        # individual
        bayesian_samples_no_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_no_pooling.mat")
                )
        beta_i = bayesian_samples_no_pooling["beta"]
        beta_i_t = beta_i.transpose((2, 0, 1, 3))
        beta_i_t_r = np.reshape(beta_i_t, (n_agents * n_samples * n_chains, n_conditions))
        h1 = plot_individual_heatmaps(beta_i_t_r, colors)
        h1.savefig(os.path.join(fig_dir, f"13_sensitivity_no_pooling_individual_bayesian.pdf"))

    if stages['plot_model_checks']:
        ## eta
        # full pooling
        # group
        bayesian_samples_full_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_full_pooling.mat")
                    )
        eta_group = bayesian_samples_full_pooling["eta"]
        fig, ax = plt.subplots(1, 1)
        for condition in range(n_conditions):
            for chain in range(n_chains):
                plt.plot(range(len(eta_group[chain,:,condition].flatten()), eta_group[chain,:,condition].flatten()), label = f'Chain: {chain}, condition: {condition}')
        fig.savefig(fig_dir,'14_riskaversion_modelchecks_full_pooling_group_bayesian.png')

        # partial pooling
        # group
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                )
        eta_group = bayesian_samples_partial_pooling["eta_g"]
        fig, ax = plt.subplots(1, 1)
        for condition in range(n_conditions):
            for chain in range(n_chains):
                plt.plot(range(len(eta_group[chain,:,condition].flatten()), eta_group[chain,:,condition].flatten()), label = f'Chain: {chain}, condition: {condition}')
        fig.savefig(fig_dir,'15_riskaversion_modelchecks_partial_pooling_group_bayesian.png')

        #individual
        eta_i = bayesian_samples_partial_pooling["eta"]
        for agent in range(n_agents):
            fig, ax = plt.subplots(1,1)
            for condition in range(n_conditions):
                for chain in range(n_chains):
                    plt.plot(range(len(eta_i[chain,:,agent,condition].flatten()), eta_i[chain,:,agent,condition].flatten()), label = f'Chain: {chain}, condition: {condition}')
            fig.savefig(os.path.join(fig_dir, f"16_{agent}_riskaversion_modelchecks_partial_pooling_individual_bayesian.pdf"))

        # no pooling
        # individual
        bayesian_samples_no_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_no_pooling.mat")
                )
        eta_i = bayesian_samples_no_pooling["eta"]
        for agent in range(n_agents):
            fig, ax = plt.subplots(1,1)
            for condition in range(n_conditions):
                for chain in range(n_chains):
                    plt.plot(range(len(eta_i[chain,:,agent,condition].flatten()), eta_i[chain,:,agent,condition].flatten()), label = f'Chain: {chain}, condition: {condition}')
            fig.savefig(os.path.join(fig_dir, f"17_{agent}_riskaversion_modelchecks_no_pooling_individual_bayesian.pdf"))
