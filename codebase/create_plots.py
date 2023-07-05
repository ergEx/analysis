#%% # -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from .utils import plot_individual_heatmaps, plot_single_kde, read_Bayesian_output, jasp_like_correlation, jasp_like_raincloud


def main(config_file):
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["plots"]["run"]:
        return

    print(f"\nPLOTTING")

    data_dir = config["data directoty"]
    fig_dir = config["figure directoty"]

    # If folders do not exist - create them:
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

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
    # Set slightly larger fontscale throughout, but keeping matplotlib settings
    cm = 1/2.54  # centimeters in inches (for plot size conversion)
    sns.set(font_scale=1.2, rc=rcParamsDefault)
    # params = {"font.family" : "serif", If the need occurs to set fonts
    #          "font.serif" : ["Computer Modern Serif"]}
    # plt.rcParams.update(params)

    if stages['plot_passive']:
        if data_type != 'real_data':
            print('There is no passive trajectories for simulated data')
        else:
            df_passive = pd.read_csv(os.path.join(data_dir, "all_passive_phase_data.csv"), sep="\t")
            fig, ax = plt.subplots(1,2, figsize=(23 * cm, 4.75 * cm))
            ax = ax.flatten()
            for c, con in enumerate(set(df_passive.eta)):
                tmp_df = df_passive.query("eta == @con")
                pivoted_df = tmp_df.pivot(index='trial', columns='participant_id', values='wealth')
                for i, participant in enumerate(pivoted_df.columns):
                    ax[c].plot(pivoted_df.index, pivoted_df[participant],  color = colors[i])
                ax[c].set(title = title_dict[c],xlabel="Trial", ylabel="Wealth")
                if c == 1:
                    ax[c].set(yscale="log", ylabel="Wealth")

                for i in [45, 45 * 2]:
                    ax[c].axvline(x=i, linestyle="--", color="grey")
                ax[c].plot([], linestyle="--", color="grey", label='reset')
                ax[c].legend(loc="upper left")

            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, '01_passive_trajectories.png'), dpi=600, bbox_inches='tight')

    if stages['plot_no_brainers']:
        if data_type != "real_data":
            print('There is no no-brainer data for simulated data')
        else:
            df_no_brainer = pd.read_csv(os.path.join(data_dir, "all_no_brainer_data.csv"), sep="\t")
            fig, ax = plt.subplots(1,2, figsize=(12 * cm, 7 * cm))
            ax = ax.flatten()
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
                sns.stripplot(
                    x="trial_bins",
                    y="response_correct",
                    hue="participant_id",
                    palette=colors,
                    data=df_prop,
                    ax=ax[c],
                    s=7
                )
                ax[c].set(
                    ylim=(0, 1), ylabel="No-brainers: Proportion correct", xlabel="", title=title_dict[c]
                )
                if c == 1:
                    ax[c].set(ylabel='')

                #ax[c].collections[0].set_sizes([75])
                ax[c].legend().remove()
                ax[c].axhline(y=0.8, color="black", linestyle="--")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, '02_no_brainers.png'), dpi=600, bbox_inches='tight')

    if stages['plot_active']:
        if data_type != 'real_data':
            print('There is no passive trajectories for simulated data')
        else:
            df_active = pd.read_csv(os.path.join(data_dir, "all_active_phase_data.csv"), sep="\t")
            fig, ax = plt.subplots(1,2, figsize=(23 * cm, 4.75 * cm))
            ax = ax.flatten()
            for c, con in enumerate(set(df_active.eta)):
                tmp_df = df_active.query("eta == @con")
                pivoted_df = tmp_df.pivot(index='trial', columns='participant_id', values='wealth')
                for i, participant in enumerate(pivoted_df.columns):
                    ax[c].plot(pivoted_df.index, pivoted_df[participant], color = colors[i])
                ax[c].set(title=title_dict[c],xlabel="Trial", ylabel="Wealth")
                if c == 1:
                    ax[c].set(yscale="log", ylabel="Wealth")

                ax[c].axhline(soft_limits[c][1], linestyle="--", color="grey", label='upper limit')
                ax[c].axhline(soft_limits[c][0], linestyle="--", color="grey", label='lower limit')
                #ax[c].legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, '03_active_trajectories.png'), dpi=600, bbox_inches='tight')

    if stages['plot_riskaversion_bracketing']:
        #Full pooling
        bracketing_overview = pd.read_csv(os.path.join(data_dir, "bracketing_overview.csv"), sep = '\t')

        df_tmp = bracketing_overview[bracketing_overview.participant == 'all']
        add = np.random.normal(df_tmp[df_tmp.dynamic == 0.0].log_reg_decision_boundary, df_tmp[df_tmp.dynamic == 0.0].log_reg_std_dev, n_samples * n_conditions)
        mul = np.random.normal(df_tmp[df_tmp.dynamic == 1.0].log_reg_decision_boundary, df_tmp[df_tmp.dynamic == 1.0].log_reg_std_dev, n_samples * n_chains)
        fig, ax = plt.subplots(1, 1)
        ax = plot_single_kde([add,mul], ax, x_fiducials=[0, 1])
        fig.savefig(os.path.join(fig_dir, '04_riskaversion_full_pooling_group_bracketing.pdf'))

        #No pooling
        df_tmp = bracketing_overview[bracketing_overview.participant != 'all']
        etas = np.empty([n_agents,n_samples*n_chains,n_conditions])
        for i, participant in enumerate(list(set(df_tmp.participant))):
            for c, con in enumerate(list(set(df_tmp.dynamic))):
                tmp_df = df_tmp.query('participant == @participant and dynamic == @con')
                if float(tmp_df.log_reg_std_dev) <= 0:
                    continue
                etas[i,:,c] = np.random.normal(tmp_df.log_reg_decision_boundary, tmp_df.log_reg_std_dev, n_samples*n_chains)
        etas_log_r = np.reshape(etas, (n_agents * n_samples * n_chains, n_conditions))
        h1 = plot_individual_heatmaps(etas_log_r, colors, hue = np.repeat(np.arange(n_agents), n_chains * n_samples),
                                      x_fiducial=[0], y_fiducial=[1])
        h1.savefig(os.path.join(fig_dir, '05_riskaversion_no_pooling_individual_bracketing.pdf'))

    if stages['plot_riskaversion_bayesian']:
        # full pooling
        # group
        bayesian_samples_full_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_full_pooling.mat")
                    )
        eta_group = bayesian_samples_full_pooling["eta_g"]
        fig, ax = plt.subplots(1, 1)
        ax = plot_single_kde([eta_group[:,:,0].flatten(),eta_group[:,:,1].flatten()], ax, x_fiducials=[0, 1])
        fig.savefig(os.path.join(fig_dir,'06_riskaversion_full_pooling_group_bayesian.pdf'))

        # partial pooling
        # group
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                )
        eta_group = bayesian_samples_partial_pooling["eta_g"]
        fig, ax = plt.subplots(1, 1)
        ax = plot_single_kde([eta_group[:,:,0].flatten(),eta_group[:,:,1].flatten()], ax, x_fiducials=[0, 1])
        fig.savefig(os.path.join(fig_dir,'07_riskaversion_partial_pooling_group_bayesian.pdf'))

        #individual
        eta_i = bayesian_samples_partial_pooling["eta_i"]
        eta_i_part_t = eta_i.transpose((2, 0, 1, 3))
        eta_i_part_t_r = np.reshape(eta_i_part_t, (n_agents * n_samples * n_chains, n_conditions))
        h1 = plot_individual_heatmaps(eta_i_part_t_r, colors, hue = np.repeat(np.arange(n_agents), n_chains * n_samples),
                                      x_fiducial=[0], y_fiducial=[1])
        h1.savefig(os.path.join(fig_dir, f"08_riskaversion_partial_pooling_individual_bayesian.pdf"))

        # no pooling
        # individual
        bayesian_samples_no_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_no_pooling.mat")
                )
        eta_i = bayesian_samples_no_pooling["eta_i"]
        eta_i_t = eta_i.transpose((2, 0, 1, 3))
        eta_i_t_r = np.reshape(eta_i_t, (n_agents * n_samples * n_chains, n_conditions))
        h1 = plot_individual_heatmaps(eta_i_t_r, colors,  hue = np.repeat(np.arange(n_agents), n_chains * n_samples),
                                      x_fiducial=[0], y_fiducial=[1])
        h1.savefig(os.path.join(fig_dir, f"09_riskaversion_no_pooling_individual_bayesian.pdf"))

    if stages['plot_jasp_like']:

        jasp_data = pd.read_csv(os.path.join(data_dir, "jasp_input.csv"), sep = '\t')
        # Plotting partial_pooling
        fig, ax = jasp_like_raincloud(jasp_data, '0.0_partial_pooling', '1.0_partial_pooling')
        fig.savefig(os.path.join(fig_dir, f"10_raincloud_riskaversion_partial_pooling.pdf"), dpi=600, bbox_inches='tight')
        fig, ax = jasp_like_correlation(jasp_data, '0.0_partial_pooling', '1.0_partial_pooling' )
        fig.savefig(os.path.join(fig_dir, f"11_correlation_riskaversion_partial_pooling.pdf"), dpi=600, bbox_inches='tight')
        # Plotting no_pooling
        fig, ax = jasp_like_raincloud(jasp_data, '0.0_no_pooling', '1.0_no_pooling')
        fig.savefig(os.path.join(fig_dir, f"12_raincloud_riskaversion_no_pooling.pdf"), dpi=600, bbox_inches='tight')
        fig, ax = jasp_like_correlation(jasp_data, '0.0_no_pooling', '1.0_no_pooling')
        fig.savefig(os.path.join(fig_dir, f"13_correlation_riskaversion_no_pooling.pdf"), dpi=600, bbox_inches='tight')
        # Plotting bracketing
        fig, ax = jasp_like_raincloud(jasp_data, '0.0_bracketing', '1.0_bracketing')
        fig.savefig(os.path.join(fig_dir, f"14_raincloud_riskaversion_no_pooling.pdf"), dpi=600, bbox_inches='tight')
        fig, ax = jasp_like_correlation(jasp_data, '0.0_bracketing', '1.0_bracketing')
        fig.savefig(os.path.join(fig_dir, f"15_correlation_riskaversion_bracketing.pdf"), dpi=600, bbox_inches='tight')

    return

    if stages['plot_sensitivity_bayesian']:
        # full pooling
        # group
        bayesian_samples_full_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_full_pooling.mat")
                    )
        beta_group = bayesian_samples_full_pooling["beta_g"]
        fig, ax = plt.subplots(1, 1)
        ax = plot_single_kde([beta_group[:,:,0].flatten(),beta_group[:,:,1].flatten()], ax)
        fig.savefig(os.path.join(fig_dir,'10_sensitivity_full_pooling_group_bayesian.png'))

        # partial pooling
        # group
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                )
        beta_group = bayesian_samples_partial_pooling["beta_g"]
        fig, ax = plt.subplots(1, 1)
        ax = plot_single_kde([beta_group[:,:,0].flatten(),beta_group[:,:,1].flatten()], ax)
        fig.savefig(os.path.join(fig_dir,'11_sensitivity_partial_pooling_group_bayesian.png'))

        #individual
        beta_i = bayesian_samples_partial_pooling["beta_i"]
        beta_i_part_t = beta_i.transpose((2, 0, 1, 3))
        beta_i_part_t_r = np.reshape(beta_i_part_t, (n_agents * n_samples * n_chains, n_conditions))
        h1 = plot_individual_heatmaps(beta_i_part_t_r, colors)
        h1.savefig(os.path.join(fig_dir, f"12_sensitivity_partial_pooling_individual_bayesian.png"))

        # no pooling
        # individual
        bayesian_samples_no_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_no_pooling.mat")
                )
        beta_i = bayesian_samples_no_pooling["beta_i"]
        beta_i_t = beta_i.transpose((2, 0, 1, 3))
        beta_i_t_r = np.reshape(beta_i_t, (n_agents * n_samples * n_chains, n_conditions))
        h1 = plot_individual_heatmaps(beta_i_t_r, colors)
        h1.savefig(os.path.join(fig_dir, f"13_sensitivity_no_pooling_individual_bayesian.png"))

    if stages['plot_model_checks']:
        ## eta
        # full pooling
        # group
        bayesian_samples_full_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_full_pooling.mat")
                    )
        eta_group = bayesian_samples_full_pooling["eta_g"]
        fig, ax = plt.subplots(1, 1)
        for c in range(n_conditions):
            for chain in range(n_chains):
                plt.plot(range(len(eta_group[chain,:,c].flatten()), eta_group[chain,:,c].flatten()), label = f'Chain: {chain}, condition: {c}', alpha = 0.5)
        fig.savefig(os.path.join(fig_dir,'14_riskaversion_modelchecks_full_pooling_group_bayesian.png'))

        # partial pooling
        # group
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                )
        eta_group = bayesian_samples_partial_pooling["eta_g"]
        fig, ax = plt.subplots(1, 1)
        for c in range(n_conditions):
            for chain in range(n_chains):
                ax.plot(range(len(eta_group[chain,:,c])), eta_group[chain,:,c], label = f'Chain: {chain}, condition: {c}', alpha = 0.5)
        fig.legend()
        fig.savefig(os.path.join(fig_dir,'15_riskaversion_modelchecks_partial_pooling_group_bayesian.png'))

        #individual
        eta_i = bayesian_samples_partial_pooling["eta_i"]
        for i in range(n_agents):
            fig, ax = plt.subplots(1,1)
            for c in range(n_conditions):
                for chain in range(n_chains):
                    ax.plot(range(len(eta_i[chain,:,i,c])), eta_i[chain,:,i,c], label = f'Chain: {chain}, condition: {c}', alpha = 0.5)
            fig.legend()
            fig.savefig(os.path.join(fig_dir, f"16_{i}_riskaversion_modelchecks_partial_pooling_individual_bayesian.png"))

        # no pooling
        # individual
        bayesian_samples_no_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_no_pooling.mat")
                )
        eta_i = bayesian_samples_no_pooling["eta"]
        for i in range(n_agents):
            fig, ax = plt.subplots(1,1)
            for c in range(n_conditions):
                for chain in range(n_chains):
                    ax.plot(range(len(eta_i[chain,:,i,c])), eta_i[chain,:,i,c], label = f'Chain: {chain}, condition: {c}', alpha = 0.5)
            fig.legend()
            fig.savefig(os.path.join(fig_dir, f"17_{i}_riskaversion_modelchecks_no_pooling_individual_bayesian.png"))
