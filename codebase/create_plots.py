#%% # -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import rcParamsDefault
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "text.usetex": True})

cm = 1/2.54  # centimeters in inches (for plot size conversion)
fig_size = (6.5 * cm , 5.75 * cm)

from .utils import jasp_like_correlation, paired_swarm_plot, plot_individual_heatmaps, plot_single_kde, \
    read_Bayesian_output


def main(config_file):
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["plots"]["run"]:
        return

    print(f"\nPLOTTING")

    data_dir = config["data directory"]
    fig_dir = config["figure directory"]

    # If folders do not exist - create them:
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    data_type = config["data_type"]
    quality_dictionary = {'chains': [2,4,4,4], 'samples': [5e1,5e2,5e3,1e4,2e4], 'manual_burnin': [1e1,1e3,1e4,2e4,4e4]}
    n_agents = config["n_agents"]
    burn_in = int(quality_dictionary['manual_burnin'][config['qual'] - 1])
    n_samples = int(quality_dictionary['samples'][config['qual'] - 1] - burn_in)
    n_chains = int(quality_dictionary['chains'][config['qual'] - 1])
    n_conditions = config["n_conditions"]

    stages = config["plots"]

    title_dict={0: "Additive", 1: "Multiplicative"}
    soft_limits = {0.0: [-500, 2_500], 1.0: [64 , 15_589]}

    colors = ['blue','red']
    # Set slightly larger fontscale throughout, but keeping matplotlib settings
    sns.set_context('paper', font_scale=1.0) #, rc=rcParamsDefault)
    # params = {"font.family" : "serif", If the need occurs to set fonts
    #          "font.serif" : ["Computer Modern Serif"]}
    # plt.rcParams.update(params)

    LIMITS = [-1,2]

    if stages['plot_passive']:
        if data_type != 'real_data':
            print('There is no passive trajectories for simulated data')
        else:
            df_passive = pd.read_csv(os.path.join(data_dir, "all_passive_phase_data.csv"), sep="\t")
            participants_to_plot = df_passive['participant_id'].unique()[:10]
            fig, ax = plt.subplots(1,2, figsize=(23 * cm, 4.75 * cm))
            ax = ax.flatten()
            for c, con in enumerate(set(df_passive.eta)):
                tmp_df = df_passive.query("eta == @con")
                pivoted_df = tmp_df.pivot(index='trial', columns='participant_id', values='wealth')
                for i, participant in enumerate(participants_to_plot):
                    ax[c].plot(pivoted_df.index, pivoted_df[participant],  color = 'black', alpha = 0.8)
                ax[c].set(title = title_dict[c],xlabel="Trial", ylabel="Wealth")
                if c == 1:
                    ax[c].set(yscale="log", ylabel="Wealth")

                for i in [45, 45 * 2]:
                    ax[c].axvline(x=i, linestyle="--", color="grey")
                ax[c].plot([], linestyle="--", color="grey", label='reset')
                ax[c].legend(loc="upper left")

            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, '01_passive_trajectories.png'), dpi=600, bbox_inches='tight')

    if stages['plot_active']:
        if data_type != 'real_data':
            print('There is no passive trajectories for simulated data')
        else:
            df_active = pd.read_csv(os.path.join(data_dir, "all_active_phase_data.csv"), sep="\t")
            participants_to_plot = df_active['participant_id'].unique()[:10]
            fig, ax = plt.subplots(2,1, figsize=((23 * cm)/2, (4.75 * cm)*2))
            ax = ax.flatten()
            for c, con in enumerate(set(df_active.eta)):
                tmp_df = df_active.query("eta == @con")
                pivoted_df = tmp_df.pivot(index='trial', columns='participant_id', values='wealth')
                for i, participant in enumerate(participants_to_plot):
                    ax[c].plot(pivoted_df.index, pivoted_df[participant], color = 'black', alpha = 0.8)
                ax[c].set(title=title_dict[c],xlabel="Trial", ylabel="Wealth")
                if c == 1:
                    ax[c].set(yscale="log", ylabel="Wealth")

                ax[c].axhline(soft_limits[c][1], linestyle="--", color="grey", label='upper limit')
                ax[c].axhline(soft_limits[c][0], linestyle="--", color="grey", label='lower limit')
                #ax[c].legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, '02_active_trajectories.png'), dpi=600, bbox_inches='tight')

    #sns.set(font_scale=1.75, rc=rcParamsDefault) # Increasing scale again.
    sns.set_context('paper', font_scale=1.1) #, rc=rcParamsDefault)

    if stages['plot_riskaversion_bracketing']:
        bracketing_overview = pd.read_csv(os.path.join(data_dir, "bracketing_overview.csv"), sep = '\t')

        df_full_pooling = bracketing_overview[bracketing_overview.participant == 'all']
        df_no_pooling = bracketing_overview[bracketing_overview.participant != 'all']

        eta_add_full_pooling = np.random.normal(df_full_pooling[df_full_pooling.dynamic == 0.0].log_reg_decision_boundary, df_full_pooling[df_full_pooling.dynamic == 0.0].log_reg_std_dev, n_samples * n_conditions)
        eta_mul_full_pooling = np.random.normal(df_full_pooling[df_full_pooling.dynamic == 1.0].log_reg_decision_boundary, df_full_pooling[df_full_pooling.dynamic == 1.0].log_reg_std_dev, n_samples * n_chains)

        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax2 = ax.twinx()
        maxi = np.empty([n_conditions,n_agents,2])
        etas_no_pooling = np.empty([n_agents,n_samples*n_chains,n_conditions])
        for i, participant in enumerate(df_no_pooling['participant_id'].unique()):
            for c, con in enumerate(df_no_pooling['dynamic'].unique()):
                tmp_df = df_no_pooling.query('participant == @participant and dynamic == @con')
                if float(tmp_df.log_reg_std_dev) <= 0:
                    continue
                tmp = np.random.normal(tmp_df.log_reg_decision_boundary, tmp_df.log_reg_std_dev, n_samples*n_chains)
                etas_no_pooling[i,:,c] = tmp
                sns.kdeplot(tmp, ax = ax, color = colors[c], alpha = 0.1)
                kde = gaussian_kde(tmp)
                maxi[c,i,0] = tmp[np.argmax(kde.pdf(tmp))]
                maxi[c,i,1] = kde.pdf(maxi[c,i,0])
        sns.kdeplot(eta_add_full_pooling, ax = ax, color = colors[0], alpha = 1, label = 'Additive')
        sns.kdeplot(eta_mul_full_pooling, ax = ax, color = colors[1], alpha = 1, label = 'Multiplicative')

        ax.set(xlim = LIMITS, xlabel = r"$\eta$", ylabel = '')
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        ax.spines[['left', 'top','right']].set_visible(False)

        ax2.set(ylabel = '')
        ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        ax2.spines[['left', 'top', 'right']].set_visible(False)
        ax2.legend('upper right')

        fig.savefig(os.path.join(fig_dir, '03_riskaversion_bracketing_1.pdf'), dpi=600, bbox_inches='tight')


        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        sns.kdeplot(x=etas_no_pooling[:,:,0].ravel(), y=etas_no_pooling[:,:,1].ravel(), cmap="YlOrBr", fill=True, ax = ax)

        sns.lineplot(x=LIMITS, y=LIMITS, color='black', linestyle='--', ax=ax, alpha = 0.5)
        ax.axvline(0, color=colors[0], alpha=0.5, linestyle='--')
        ax.axhline(1, color=colors[1], alpha=0.5, linestyle='--')
        ax.set(xlim = LIMITS, ylim = LIMITS, xlabel = r"$\eta^{\mathrm{add}}$", ylabel = r"$\eta^{\mathrm{mul}}$")
        ax.spines[['top','right']].set_visible(False)

        ax.scatter(x=maxi[0, :, 0], y=maxi[1, :, 0], marker='x', color='black', label = 'MAP estimates')
        ax.legend('lower right')

        fig.savefig(os.path.join(fig_dir, '03_riskaversion_bracketing_2.pdf'), dpi=600, bbox_inches='tight')

    if stages['plot_riskaversion_bayesian']:
        labels = ['Additive','Multiplicative']
        #no pooling and full pooling
        bayesian_samples_full_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_full_pooling.mat")
                    )
        eta_g = bayesian_samples_full_pooling["eta_g"][:,burn_in:,:]

        bayesian_samples_no_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_no_pooling.mat")
                )
        eta_i = bayesian_samples_no_pooling["eta_i"][:,burn_in:,:,:]

        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax2 = ax.twinx()
        maxi = np.empty([n_conditions,n_agents,2])
        for c in range(n_conditions):
            for i in range(n_agents):
                data_tmp = eta_i[:,:,i,c].ravel()
                sns.kdeplot(data_tmp, ax = ax, color = colors[c], alpha = 0.1)
                kde = gaussian_kde(data_tmp)

                maxi[c,i,0] = data_tmp[np.argmax(kde.pdf(data_tmp))]
                maxi[c,i,1] = kde.pdf(maxi[c,i,0])

            sns.kdeplot(eta_g[:,:,c].ravel(), ax = ax2, color = colors[c], linestyle = '-', label = labels[c])

        ax.set(xlim = LIMITS, xlabel = r"$\eta$", ylabel = '')
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        ax.spines[['left', 'top','right']].set_visible(False)

        ax2.set(ylabel = '')
        ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        ax2.spines[['left', 'top', 'right']].set_visible(False)
        ax2.legend('upper right')

        fig.savefig(os.path.join(fig_dir, '04_riskaversion_bayesian_1.pdf'), dpi=600, bbox_inches='tight')


        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        sns.kdeplot(x=eta_i[:,:,:,0].ravel(), y=eta_i[:,:,:,1].ravel(), cmap="YlOrBr", fill=True, ax = ax)

        sns.lineplot(x=LIMITS, y=LIMITS, color='black', linestyle='--', ax=ax, alpha = 0.5)
        ax.axvline(0, color='blue', alpha=0.5, linestyle='--')
        ax.axhline(1, color='red', alpha=0.5, linestyle='--')
        ax.set(xlim = LIMITS, ylim = LIMITS, xlabel = r"$\eta^{\mathrm{add}}$", ylabel = r"$\eta^{\mathrm{mul}}$")
        ax.spines[['top','right']].set_visible(False)

        ax.scatter(x=maxi[0, :, 0], y=maxi[1, :, 0], marker='x', color='black', label = 'MAP estimates')
        ax.legend('lower right')

        fig.savefig(os.path.join(fig_dir, '04_riskaversion_bayesian_2.pdf'), dpi=600, bbox_inches='tight')

        #Partial pooling
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                    )
        eta_g = bayesian_samples_partial_pooling["eta_g"][:,burn_in:,:]
        eta_i = bayesian_samples_partial_pooling["eta_i"][:,burn_in:,:,:]

        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax2 = ax.twinx()
        maxi = np.empty([n_conditions,n_agents,2])
        for c in range(n_conditions):
            for i in range(n_agents):
                data_tmp = eta_i[:,:,i,c].ravel()
                sns.kdeplot(data_tmp, ax = ax, color = colors[c], alpha = 0.1)
                kde = gaussian_kde(data_tmp)

                maxi[c,i,0] = data_tmp[np.argmax(kde.pdf(data_tmp))]
                maxi[c,i,1] = kde.pdf(maxi[c,i,0])

            sns.kdeplot(eta_g[:,:,c].ravel(), ax = ax2, color = colors[c], linestyle = '-', label = labels[c])

        ax.set(xlim = LIMITS, xlabel = r"$\eta$", ylabel = '')
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        ax.spines[['left', 'top','right']].set_visible(False)

        ax2.set(ylabel = '')
        ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        ax2.spines[['left', 'top', 'right']].set_visible(False)
        ax2.legend('upper right')

        fig.savefig(os.path.join(fig_dir, '04_riskaversion_bayesian_3.pdf'), dpi=600, bbox_inches='tight')


        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        sns.kdeplot(x=eta_i[:,:,:,0].ravel(), y=eta_i[:,:,:,1].ravel(), cmap="YlOrBr", fill=True, ax = ax)

        sns.lineplot(x=LIMITS, y=LIMITS, color='black', linestyle='--', ax=ax, alpha = 0.5)
        ax.axvline(0, color='blue', alpha=0.5, linestyle='--')
        ax.axhline(1, color='red', alpha=0.5, linestyle='--')
        ax.set(xlim = LIMITS, ylim = LIMITS, xlabel = r"$\eta^{\mathrm{add}}$", ylabel = r"$\eta^{\mathrm{mul}}$")
        ax.spines[['top','right']].set_visible(False)

        ax.scatter(x=maxi[0, :, 0], y=maxi[1, :, 0], marker='x', color='black', label = 'MAP estimates')
        ax.legend('lower right')

        fig.savefig(os.path.join(fig_dir, '04_riskaversion_bayesian_4.pdf'), dpi=600, bbox_inches='tight')

    if stages['plot_mcmc_samples']:
        # full pooling
        # group
        bayesian_samples_full_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_full_pooling.mat")
                    )
        eta_g = bayesian_samples_full_pooling["eta_g"]

        fig, ax = plt.subplots(1, 1, figsize=(23 * cm, 4.75 * cm))
        ax.axvspan(0, burn_in, alpha=0.2, color='grey')
        for c in range(n_chains):
            ax.plot(range(len(eta_g[c,:,0].ravel())), eta_g[c,:,0].ravel(), alpha = (c+1)/n_chains, color = colors[0])
            ax.plot(range(len(eta_g[c,:,1].ravel())), eta_g[c,:,1].ravel(), alpha = (c+1)/n_chains, color = colors[1])
        ax.set_xlim(left = 0)
        ax.legend(['Burn in', 'Additive', 'Multiplicative'], loc = 'upper right')
        ax.set(xlabel="Samples", ylabel=f"$\eta$")
        fig.savefig(os.path.join(fig_dir, '05_riskaversion_mcmc_samples_1.png'), dpi=600, bbox_inches='tight')


        # partial pooling
        # group
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                )
        eta_group = bayesian_samples_partial_pooling["eta_g"]

        fig, ax = plt.subplots(1, 1, figsize=(23 * cm, 4.75 * cm))
        ax.axvspan(0, burn_in, alpha=0.2, color='grey', label = 'Burn in')
        for c in range(n_chains):
            ax.plot(range(len(eta_g[c,:,0].ravel())), eta_g[c,:,0].ravel(), alpha = (c+1)/n_chains, color = colors[0])
            ax.plot(range(len(eta_g[c,:,1].ravel())), eta_g[c,:,1].ravel(), alpha = (c+1)/n_chains, color = colors[1])
        ax.set_xlim(left = 0)
        ax.legend(['Burn in', 'Additive', 'Multiplicative'], loc = 'upper right')
        ax.set(xlabel="Samples", ylabel=f"$\eta$")
        fig.savefig(os.path.join(fig_dir, '05_riskaversion_mcmc_samples_2.png'), dpi=600, bbox_inches='tight')

    if stages['plot_sensitivity_bayesian']:
        labels = ['Additive','Multiplicative']
        #no pooling and full pooling
        bayesian_samples_full_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_full_pooling.mat")
                    )
        beta_g = bayesian_samples_full_pooling["beta_g"][:,burn_in:,:]

        bayesian_samples_no_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_no_pooling.mat")
                )
        beta_i = bayesian_samples_no_pooling["beta_i"][:,burn_in:,:,:]

        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax2 = ax.twinx()
        beta_maxi = np.empty([n_conditions,n_agents])
        for c in range(n_conditions):
            for i in range(n_agents):
                data_tmp = np.log(beta_i[:,:,i,c].ravel())
                sns.kdeplot(data_tmp, ax = ax, color = colors[c], alpha = 0.1)
                kde = gaussian_kde(data_tmp)

                beta_maxi[c,i] = data_tmp[np.argmax(kde.pdf(data_tmp))]

            sns.kdeplot(np.log(beta_g[:,:,c].ravel()), ax = ax2, color = colors[c], linestyle = '-')

        ax.set( xlabel = r"$\ln \beta$", ylabel = '')
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        ax.spines[['left', 'top','right']].set_visible(False)

        ax2.set(ylabel = '')
        ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        ax2.spines[['left', 'top', 'right']].set_visible(False)

        fig.savefig(os.path.join(fig_dir, '06_sensitivity_bayesian_1.png'), dpi=600, bbox_inches='tight')

        #partial pooling
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                    )
        beta_g = bayesian_samples_partial_pooling["beta_g"][:,burn_in:,:]
        beta_i = bayesian_samples_partial_pooling["beta_i"][:,burn_in:,:,:]

        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax2 = ax.twinx()
        beta_maxi = np.empty([n_conditions,n_agents])
        for c in range(n_conditions):
            for i in range(n_agents):
                data_tmp = np.log(beta_i[:,:,i,c].ravel())
                sns.kdeplot(data_tmp, ax = ax, color = colors[c], alpha = 0.1)
                kde = gaussian_kde(data_tmp)

                beta_maxi[c,i] = data_tmp[np.argmax(kde.pdf(data_tmp))]

            sns.kdeplot(np.log(beta_g[:,:,c].ravel()), ax = ax2, color = colors[c], linestyle = '-')

        ax.set( xlabel = r"$\ln \beta$", ylabel = '')
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        ax.spines[['left', 'top','right']].set_visible(False)

        ax2.set(ylabel = '')
        ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        ax2.spines[['left', 'top', 'right']].set_visible(False)

        fig.savefig(os.path.join(fig_dir, '06_sensitivity_bayesian_2png'), dpi=600, bbox_inches='tight')

    if stages['plot_model_comparison']:
        #no pooling
        #EUT v EE
        df = pd.read_csv(os.path.join(data_dir, "proportions_data_no_pooling_1.csv"), sep="\t")
        fig, ax = plt.subplots(1, 2, figsize=(23 * cm, 4.75 * cm), gridspec_kw={'width_ratios': [8, 2]})

        sns.heatmap(df[['EUT','EE']], cmap='gray_r', xticklabels=False, yticklabels=['EUT', 'EE'], cbar=False, ax=ax[0])
        ax[0].set_xlabel('Participants')

        ax[1].bar(['EUT', 'EE'], df[['EUT', 'EE']].sum() / df[['EUT', 'EE']].sum().sum(), color='black')

        fig.savefig(os.path.join(fig_dir, '07_model_selection_1.png.png'), dpi=600, bbox_inches='tight')

        #EUT v EE2
        df = pd.read_csv(os.path.join(data_dir, "proportions_data_no_pooling_2.csv"), sep="\t")
        fig, ax = plt.subplots(1, 2, figsize=(23 * cm, 4.75 * cm), gridspec_kw={'width_ratios': [8, 2]})

        sns.heatmap(df[['EUT','Weak_EE']], cmap='gray_r', xticklabels=False, yticklabels=['EUT', 'Weak EE'], cbar=False, ax=ax[0])
        ax[0].set_xlabel('Participants')

        ax[1].bar(['EUT', 'Weak_EE'], df[['EUT', 'EE']].sum() / df[['EUT', 'EE']].sum().sum(), color='black')

        fig.savefig(os.path.join(fig_dir, '07_model_selection_2.png.png'), dpi=600, bbox_inches='tight')

        #Partial pooling
        #EUT v EE
        df = pd.read_csv(os.path.join(data_dir, "proportions_data_partial_pooling_1.csv"), sep="\t")
        fig, ax = plt.subplots(1, 2, figsize=(23 * cm, 4.75 * cm), gridspec_kw={'width_ratios': [8, 2]})

        sns.heatmap(df[['EUT','EE']], cmap='gray_r', xticklabels=False, yticklabels=['EUT', 'EE'], cbar=False, ax=ax[0])
        ax[0].set_xlabel('Participants')

        ax[1].bar(['EUT', 'EE'], df[['EUT', 'EE']].sum() / df[['EUT', 'EE']].sum().sum(), color='black')

        fig.savefig(os.path.join(fig_dir, '07_model_selection_3.png.png'), dpi=600, bbox_inches='tight')

        #EUT v EE2
        df = pd.read_csv(os.path.join(data_dir, "proportions_data_partial_pooling_2.csv"), sep="\t")
        fig, ax = plt.subplots(1, 2, figsize=(23 * cm, 4.75 * cm), gridspec_kw={'width_ratios': [8, 2]})

        sns.heatmap(df[['EUT','Weak_EE']], cmap='gray_r', xticklabels=False, yticklabels=['EUT', 'Weak EE'], cbar=False, ax=ax[0])
        ax[0].set_xlabel('Participants')

        ax[1].bar(['EUT', 'Weak_EE'], df[['EUT', 'EE']].sum() / df[['EUT', 'EE']].sum().sum(), color='black')

        fig.savefig(os.path.join(fig_dir, '07_model_selection_4.png.png'), dpi=600, bbox_inches='tight')



    return
