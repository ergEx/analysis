#%% # -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from tqdm.auto import tqdm

from .plotting_functions import jasp_like_raincloud, model_select_plot, posterior_dist_2dplot, posterior_dist_plot
from .support_figures.plot_nobrainer_performance import plot_nobrainers
from .utils import read_Bayesian_output

plt.rcParams.update({
    "text.usetex": True})

cm = 1/2.54  # centimeters in inches (for plot size conversion)
fig_size = (6.5 * cm , 5.75 * cm)


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
    quality_dictionary = {'chains': [2,4,4,4], 'samples': [5e1,5e2,5e3,1e4,2e4], 'manual_burnin': [1e1,1e3,1e3,2e4,4e4]}
    n_agents = config["n_agents"]
    burn_in = int(quality_dictionary['manual_burnin'][config['qual'] - 1])
    n_samples = int(quality_dictionary['samples'][config['qual'] - 1])
    n_chains = int(quality_dictionary['chains'][config['qual'] - 1])
    n_conditions = config["n_conditions"]

    stages = config["plots"]

    title_dict={0: "Additive", 1: "Multiplicative"}
    soft_limits = {0.0: [-500, 2_500], 1.0: [64 , 15_589]}

    colors    = [np.array([0, 0, 1, 1], dtype=float), np.array([1, 0, 0, 1], dtype=float)]
    colors_alpha = [np.array([0, 0, 1, 0.2], dtype=float), np.array([1, 0, 0, 0.2], dtype=float)]

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
            fig.savefig(os.path.join(fig_dir, '01_passive_trajectories.pdf'), dpi=600, bbox_inches='tight')

    if stages['plot_nobrainers']:
        fig, axes = plot_nobrainers(config, fig_size=fig_size)
        fig.savefig(os.path.join(fig_dir, '01_nobrainer_trajectories.pdf'), dpi=600, bbox_inches='tight')

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
            fig.savefig(os.path.join(fig_dir, '02_active_trajectories.pdf'), dpi=600, bbox_inches='tight')

    #sns.set(font_scale=1.75, rc=rcParamsDefault) # Increasing scale again.
    sns.set_context('paper', font_scale=1.1) #, rc=rcParamsDefault)

    if stages['plot_risk_aversion_bracketing']:

        bracketing_overview = pd.read_csv(os.path.join(data_dir, "bracketing_overview.csv"), sep = '\t')

        df_full_pooling = bracketing_overview[bracketing_overview.participant == 'all']
        df_no_pooling = bracketing_overview[bracketing_overview.participant != 'all']

        eta_g = np.zeros((n_chains, n_samples, n_conditions))
        eta_i = np.zeros((n_chains, n_samples, n_agents, n_conditions))

        for ch in range(n_chains):
            for c, con in enumerate(df_no_pooling['dynamic'].unique()):
                tmp_df_g = df_full_pooling.query('dynamic == @con')
                eta_g[ch,:,c] = np.random.normal(tmp_df_g.log_reg_decision_boundary, tmp_df_g.log_reg_std_dev, n_samples)

                for i, participant in tqdm(enumerate(df_no_pooling['participant'].unique()),
                                           total=len(df_no_pooling['participant'].unique()),
                                           desc='Approximating eta for bracketing'):
                    tmp_df_i = df_no_pooling.query('participant == @participant and dynamic == @con')
                    eta_i[ch,:,i,c] = np.random.normal(tmp_df_i.log_reg_decision_boundary, tmp_df_i.log_reg_std_dev, n_samples)

        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        labels = ['Additive','Multiplicative']

        fig, ax, ax2, maxi = posterior_dist_plot(fig, ax, eta_i, eta_g, colors, colors_alpha, n_conditions, n_agents, labels, LIMITS, r"$\eta$")

        fig.savefig(os.path.join(fig_dir, '03_riskaversion_bracketing_1.pdf'), dpi=600, bbox_inches='tight')


        fig, ax = plt.subplots(1, 1, figsize=fig_size)

        fig, ax = posterior_dist_2dplot(fig, ax, eta_i, colors_alpha, LIMITS, maxi)

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

        fig, ax, ax2, maxi = posterior_dist_plot(fig, ax, eta_i, eta_g, colors, colors_alpha, n_conditions, n_agents, labels, LIMITS, r"$\eta$")

        fig.savefig(os.path.join(fig_dir, '04_riskaversion_bayesian_1.pdf'), dpi=600, bbox_inches='tight')


        fig, ax = plt.subplots(1, 1, figsize=fig_size)

        fig, ax = posterior_dist_2dplot(fig, ax, eta_i, colors_alpha, LIMITS, maxi)

        fig.savefig(os.path.join(fig_dir, '04_riskaversion_bayesian_2.pdf'), dpi=600, bbox_inches='tight')

        #Partial pooling
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                    )
        eta_g = bayesian_samples_partial_pooling["eta_g"][:,burn_in:,:]
        eta_i = bayesian_samples_partial_pooling["eta_i"][:,burn_in:,:,:]

        fig, ax = plt.subplots(1, 1, figsize=fig_size)

        fig, ax, ax2, maxi = posterior_dist_plot(fig, ax, eta_i, eta_g, colors, colors_alpha, n_conditions, n_agents, labels, LIMITS, r"$\eta$")

        fig.savefig(os.path.join(fig_dir, '04_riskaversion_bayesian_3.pdf'), dpi=600, bbox_inches='tight')


        fig, ax = plt.subplots(1, 1, figsize=fig_size)

        fig, ax = posterior_dist_2dplot(fig, ax, eta_i, colors_alpha, LIMITS, maxi)

        fig.savefig(os.path.join(fig_dir, '04_riskaversion_bayesian_4.pdf'), dpi=600, bbox_inches='tight')

    if stages['plot_pairwise']:
        jasp_data = pd.read_csv(os.path.join(data_dir, 'jasp_input.csv'),
                                             sep='\t')

        # Hypothesis 1
        main_comparison = config['bayesfactor_analysis']['target']

        fig, axes = jasp_like_raincloud(jasp_data, f'd_h0_{main_comparison}',
                                        f'd_h1_{main_comparison}', fig_size=np.array(fig_size) * 2)

        axes[0].set(ylabel='Distance', xticklabels=['EUT', 'EE'])
        axes[2].set(xlabel='Distance EUT - EE')
        fig.savefig(os.path.join(fig_dir, '08_q1_pairwise_diff.pdf'), dpi=600, bbox_inches='tight')

        fig, axes = jasp_like_raincloud(jasp_data, f'0.0_{main_comparison}',
                                        f'1.0_{main_comparison}', fig_size=np.array(fig_size) * 2)

        axes[0].set(ylabel='Risk aversion parameter',
                    xticklabels=['Additive', 'Multiplicative'])
        axes[2].set(xlabel='$\eta$ Additive - Multiplicative')

        fig.savefig(os.path.join(fig_dir, '08_q2_pairwise_diff.pdf'), dpi=600, bbox_inches='tight')


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
            ax.plot(range(len(eta_g[c,:,0].ravel())), eta_g[c,:,0].ravel(), color = colors_alpha[0])
            ax.plot(range(len(eta_g[c,:,1].ravel())), eta_g[c,:,1].ravel(), color = colors_alpha[1])
        ax.set_xlim(left = 0)
        ax.legend(['Burn in', 'Additive', 'Multiplicative'], loc = 'upper right')
        ax.set(xlabel="Samples", ylabel=f"$\eta$")
        fig.savefig(os.path.join(fig_dir, '05_riskaversion_mcmc_samples_1.pdf'), dpi=600, bbox_inches='tight')


        # partial pooling
        # group
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                )
        eta_g = bayesian_samples_partial_pooling["eta_g"]

        fig, ax = plt.subplots(1, 1, figsize=(23 * cm, 4.75 * cm))
        ax.axvspan(0, burn_in, alpha=0.2, color='grey', label = 'Burn in')
        for c in range(n_chains):
            ax.plot(range(len(eta_g[c,:,0].ravel())), eta_g[c,:,0].ravel(), color = colors_alpha[0])
            ax.plot(range(len(eta_g[c,:,1].ravel())), eta_g[c,:,1].ravel(), color = colors_alpha[1])
        ax.set_xlim(left = 0)
        ax.legend(['Burn in', 'Additive', 'Multiplicative'], loc = 'upper right')
        ax.set(xlabel="Samples", ylabel=f"$\eta$")
        fig.savefig(os.path.join(fig_dir, '05_riskaversion_mcmc_samples_2.pdf'), dpi=600, bbox_inches='tight')

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

        fig, ax, ax2, maxi = posterior_dist_plot(fig, ax, beta_i, beta_g, colors, colors_alpha, n_conditions, n_agents, labels, LIMITS, r"$\eta$")

        fig.savefig(os.path.join(fig_dir, '06_sensitivity_bayesian_1.pdf'), dpi=600, bbox_inches='tight')

        #partial pooling
        bayesian_samples_partial_pooling = read_Bayesian_output(
                    os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling.mat")
                    )
        beta_g = bayesian_samples_partial_pooling["beta_g"][:,burn_in:,:]
        beta_i = bayesian_samples_partial_pooling["beta_i"][:,burn_in:,:,:]

        fig, ax = plt.subplots(1, 1, figsize=fig_size)

        fig, ax, ax2, maxi = posterior_dist_plot(fig, ax, beta_i, beta_g, colors, colors_alpha, n_conditions, n_agents, labels, LIMITS, r"$\beta$")

        fig.savefig(os.path.join(fig_dir, '06_sensitivity_bayesian_2.pdf'), dpi=600, bbox_inches='tight')

    if stages['plot_model_selection']:
        model_specs = {'EUT v EE' :
                            {'name': 'EUT_EE',
                            'models' : ['EUT','EE']},
                        'EUT v Weak EE' :
                            {'name': 'EUT_EE2',
                            'models' : ['EUT','Weak EE']}}

        for m, typ in enumerate(model_specs):
            model = read_Bayesian_output(
                    os.path.join(data_dir, f"Bayesian_JAGS_model_selection_{model_specs[typ]['name']}{model_specs[typ]['model_selection_type']}.mat")
                    )
            z = model['samples']['z'][:,burn_in:,:]

            fig, ax = model_select_plot(z,model_specs[typ]['models'])

            fig.savefig(os.path.join(fig_dir, f'07_model_selection_{m}.pdf'), dpi=600, bbox_inches='tight')

    if stages['plot_pooling_selection']:
        model_specs = {'data pooling' :
                            {'name': 'data_pooling',
                            'models' : ['No pooling','Partial pooling','Full pooling']}}

        model = read_Bayesian_output(
                os.path.join(data_dir, f"Bayesian_JAGS_model_selection_{model_specs['name']}{model_specs['model_selection_type']}_2.mat")
                )
        z = model['samples']['z'][:,burn_in:,:]

        fig, ax = model_select_plot(z,model_specs['models'], individual = False)

        fig.savefig(os.path.join(fig_dir, f'08_model_selection_{m}.pdf'), dpi=600, bbox_inches='tight')
    return

# %%
