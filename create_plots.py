#%% # -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

#%% read in the data
if data_type == 'real_data':
    df_passive = pd.read_csv(os.path.join(data_dir, "all_passive_phase_data.csv"), sep="\t")
    #df_no_brainer = pd.read_csv(os.path.join(data_dir, "all_no_brainer_data.csv"), sep="\t")
    df_active = pd.read_csv(os.path.join(data_dir, "all_active_phase_data.csv"), sep="\t")
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in np.linspace(0, 1, 11)]


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

# %%
