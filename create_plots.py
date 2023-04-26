#%% # -*- coding: utf-8 -*-
import os

import mat73
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

#%% Variables from config file
data_type = "0_simulation"
data_variant = 'varying_variance'
data_dir = "./data/0_simulation/varying_variance"
fig_dir = "./figs/0_simulation/varying_variance"

data_dir = "data/1_pilot"
fig_dir = "figs/1_pilot"

# This section specifies data variables
data_type = "real_data"
data_variant = "1_pilot"

title_dict={0: "Additive", 1: "Multiplicative"}

soft_limits = {0.0: [-500, 2_500], 1.0: [64 , 15_589]}

n_agents = 11

cmap = plt.get_cmap("tab20")
colors = [cmap(i) for i in np.linspace(0, 1, n_agents)]

#%% read in the data
if data_type == 'real_data':
    df_passive = pd.read_csv(os.path.join(data_dir, "all_passive_phase_data.csv"), sep="\t")
    df_no_brainer = pd.read_csv(os.path.join(data_dir, "all_no_brainer_data.csv"), sep="\t")
    df_active = pd.read_csv(os.path.join(data_dir, "all_active_phase_data.csv"), sep="\t")



#%% plot passive trajectories
if data_type == 'real_data':
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
    fig.show()

#%% plot nobrainers
if data_type == "real_data":
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
    fig.show()




#%% plot active trajectories
if data_type == 'real_data':
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
    fig.show()

#%%
def plot_single_kde(data, ax, limits = [-3,4], colors = ['blue', 'red'], labels = ['Additive', 'Multiplicative']):
    maxi = np.empty([2,2])
    for i in range(2):
        sns.kdeplot(data[i], color=colors[i], label=labels[i], fill=True, ax=ax)

        kde = gaussian_kde(data[i])

        maxi[i,0] = data[i][np.argmax(kde.pdf(data[i]))]
        maxi[i,1] = kde.pdf(maxi[i,0])

    ax.axvline(maxi[0,0], ymax=maxi[0,1] / (ax.get_ylim()[1]), color='black', linestyle='--')
    ax.axvline(maxi[1,0], ymax=maxi[1,1] / (ax.get_ylim()[1]), color='black', linestyle='--')
    ax.plot([], ls="--", color="black", label="Estimates")
    ax.legend(loc="upper left")
    ax.set(
        title="",
        xlabel="Riskaversion parameter",
        ylabel="",
        xlim=limits,
        yticks=[],
        xticks=np.linspace(limits[0], limits[1], limits[1]-limits[0]+1)
    )
    return ax
# %% Bracketing method risk aversion parameter
bracketing_overview = pd.read_csv(os.path.join(data_dir, "bracketing_overview.csv"), sep = '\t')

#Full pooling
df = bracketing_overview[bracketing_overview.participant == 'all']
add = np.random.normal(df[df.dynamic == 0.0].log_reg_decision_boundary, df[df.dynamic == 0.0].log_reg_std_dev, 5000 * 4)
mul = np.random.normal(df[df.dynamic == 1.0].log_reg_decision_boundary, df[df.dynamic == 1.0].log_reg_std_dev, 5000 * 4)
fig, ax = plt.subplots(1, 1)
ax = plot_single_kde([add,mul], ax)
plt.show()

#%%
def read_Bayesian_output(file_path: str) -> dict:
    """Read HLM output file.

    Args:
        Filepath to where the Bayesian output is found

    Returns:
        dict: Dictionary containing the HLM samples.

    """
    mat = mat73.loadmat(file_path)
    return mat["samples"]

#%% #bayesian method risk aversion parameter
# partial pooling
bayesian_samples_partial_pooling = read_Bayesian_output(
            os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_pooling_individuals.mat")
        )
eta_group = bayesian_samples_partial_pooling["eta_g"]
fig, ax = plt.subplots(1, 1)
ax = plot_single_kde([eta_group[:,:,0].flatten(),eta_group[:,:,1].flatten()], ax)
plt.show()

# full pooling
bayesian_samples_full_pooling = read_Bayesian_output(
            os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_group.mat")
            )
eta_group = bayesian_samples_full_pooling["eta"]
fig, ax = plt.subplots(1, 1)
ax = plot_single_kde([eta_group[:,:,0].flatten(),eta_group[:,:,1].flatten()], ax)
plt.show()
# %%
def plot_individual_heatmaps(data, colors, limits = [-3,4], hue = np.repeat(np.arange(n_agents), 4 * 5000)):
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

    h1.set_axis_labels("Additive condition", "Multiplicative condition")
    h1.ax_joint.set_xticks(np.linspace(limits[0], limits[1], limits[1]-limits[0]+1))
    h1.ax_joint.set_yticks(np.linspace(limits[0], limits[1], limits[1]-limits[0]+1))
    sns.lineplot(x=limits, y=limits, color='black', linestyle='--', ax=h1.ax_joint)
    return h1

#%% bracketing method
df_tmp = bracketing_overview[bracketing_overview.participant != 'all']
etas = np.empty([n_agents,5000*4,2])
for i, participant in enumerate(list(set(df_tmp.participant))):
    for c, con in enumerate(list(set(df_tmp.dynamic))):
        tmp_df = df_tmp.query('participant == @participant and dynamic == @con')
        etas[i,:,c] = np.random.normal(tmp_df.log_reg_decision_boundary, tmp_df.log_reg_std_dev, 5000*4)
etas_log_r = np.reshape(etas, (n_agents * 5000 * 4, 2))
h1 = plot_individual_heatmaps(etas_log_r, colors)
plt.show()
# %% bayesian mathods
# no pooling
bayesian_samples_no_pooling = read_Bayesian_output(
            os.path.join(data_dir, "Bayesian_JAGS_parameter_estimation_no_pooling.mat")
        )
eta_i = bayesian_samples_no_pooling["eta"]
eta_i_t = eta_i.transpose((2, 0, 1, 3))
eta_i_t_r = np.reshape(eta_i_t, (11 * 5000 * 4, 2))
h1 = plot_individual_heatmaps(eta_i_t_r, colors)
h1.savefig(os.path.join(fig_dir, f"0_7_bayesian_nopool_heatmap.pdf"))

#partial pooling
eta_i = bayesian_samples_partial_pooling["eta"]
eta_i_part_t = eta_i.transpose((2, 0, 1, 3))
eta_i_part_t_r = np.reshape(eta_i_part_t, (11 * 5000 * 4, 2))
h1 = plot_individual_heatmaps(eta_i_part_t_r, colors)
h1.savefig(os.path.join(fig_dir, f"0_7_bayesian_partialpool_heatmap.pdf"))
plt.show()

# %%
