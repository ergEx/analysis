import sys
import time
import os
import yaml
import traceback
from codebase import utils
import pandas as pd
import numpy as np
import pingouin as pg

from codebase.plotting_functions import posterior_dist_2dplot, posterior_dist_plot

from codebase.utils import read_Bayesian_output

from codebase.support_figures.plot_nobrainer_performance import plot_nobrainers
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    config_file = utils.get_config_filename(sys.argv)

    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    data_type = config["data_type"]
    data_variant = config["data_variant"]

    if not os.path.isdir(config["figure directory"]):
        os.makedirs(config["figure directory"])

    start_time = time.time()
    print(f"\n--- {time.ctime(start_time)} --- ")
    print("\nSTARTING ANALYSIS")
    print(f"Data: {data_type} \nVariant: {data_variant}")

    # Now subtract condition 1 from condition 2
    labels = ["Additive", "Multiplicative"]

    LIMITS = [-1, 2]
    quality_dictionary = {
        "chains": [2, 4, 4, 4],
        "samples": [5e1, 5e2, 5e3, 1e4, 2e4],
        "manual_burnin": [1e1, 1e3, 1e3, 2e4, 4e4],
    }
    n_agents = config["n_agents"]
    burn_in = int(quality_dictionary["manual_burnin"][config["qual"] - 1])
    n_conditions = config["n_conditions"]
    colors = [np.array([0, 0, 1, 1], dtype=float), np.array([1, 0, 0, 1], dtype=float)]
    colors_alpha = [
        np.array([0, 0, 1, 0.2], dtype=float),
        np.array([1, 0, 0, 0.2], dtype=float),
    ]
    cm = 1 / 2.54  # centimeters in inches (for plot size conversion)
    fig_size = (6.5 * cm, 5.75 * cm)
    # Set slightly larger fontscale throughout, but keeping matplotlib settings
    sns.set_context("paper", font_scale=1.0)  # , rc=rcParamsDefault)

    data_dir = config["data directory"]

    _, _, df = plot_nobrainers(config, (10, 10))
    plt.close()

    grouped_nobs = (
        df[["participant_id", "run", "eta", "response_correct"]]
        .groupby(["participant_id", "run", "eta"])
        .mean()
        .reset_index()
    )

    grouped_total = grouped_nobs.groupby(["participant_id", "eta"]).mean().reset_index()
    grouped_total["run"] = "run average"

    grouped_nobs = pd.concat([grouped_nobs, grouped_total])

    nobs_wide = grouped_nobs.pivot_table(
        index=["participant_id", "run"],
        columns="eta",
        values="response_correct",
        aggfunc="first",
    )

    delta_correct = nobs_wide[1.0] - nobs_wide[0.0]
    delta_correct = delta_correct.rename("delta_correct")
    delta_correct = delta_correct.reset_index()

    fig, axes = plt.subplots(1, 1, figsize=(fig_size[0] * 1.5, fig_size[1] * 1.5))
    sns.boxplot(x="run", y="delta_correct", data=delta_correct, ax=axes)
    sns.stripplot(x="run", y="delta_correct", data=delta_correct, ax=axes)
    axes.set(ylabel=r"Proportion correct: Multiplicative - Additive")
    axes.axhline(0)
    axes.spines[["top", "right"]].set_visible(False)

    fig.savefig(
        os.path.join(config["figure directory"], "Rev1_proportion_correct.pdf"),
        dpi=600,
        bbox_inches="tight",
    )
    fig = []

    delta_correct["delta_correct"] = delta_correct["delta_correct"].astype(float)
    pg_test = pg.ttest(delta_correct.query("run == 'run average'")["delta_correct"], 0)
    pg_test.insert(0, "Run", "Average")
    pg_test["BF01"] = 1 / pg_test["BF10"].astype(float)
    pg_test["Test"] = "Difference in no-brainer performance"

    bayesian_samples_partial_pooling = read_Bayesian_output(
        os.path.join(
            data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling_split1.mat"
        )
    )
    eta_g = bayesian_samples_partial_pooling["eta_g"][:, burn_in:, :]
    eta_i = bayesian_samples_partial_pooling["eta_i"][:, burn_in:, :, :]

    fig, axes = plt.subplots(2, 2, figsize=(fig_size[0] * 2, fig_size[1] * 2))
    axes = axes.flatten()

    fig, ax, ax2, maxi = posterior_dist_plot(
        fig,
        axes[0],
        eta_i,
        eta_g,
        colors,
        colors_alpha,
        n_conditions,
        n_agents,
        labels,
        LIMITS,
        r"$\eta$",
    )
    fig, ax = posterior_dist_2dplot(fig, axes[2], eta_i, colors_alpha, LIMITS, maxi)

    bayesian_samples_partial_pooling = read_Bayesian_output(
        os.path.join(
            data_dir, "Bayesian_JAGS_parameter_estimation_partial_pooling_split2.mat"
        )
    )
    eta_g = bayesian_samples_partial_pooling["eta_g"][:, burn_in:, :]
    eta_i = bayesian_samples_partial_pooling["eta_i"][:, burn_in:, :, :]

    fig, ax, ax2, maxi = posterior_dist_plot(
        fig,
        axes[1],
        eta_i,
        eta_g,
        colors,
        colors_alpha,
        n_conditions,
        n_agents,
        labels,
        LIMITS,
        r"$\eta$",
    )

    fig, ax = posterior_dist_2dplot(fig, axes[3], eta_i, colors_alpha, LIMITS, maxi)

    axes[0].set(title="First half of data")
    axes[1].set(title="Second half of data")

    fig.savefig(
        os.path.join(config["figure directory"], "Rev2_partial_eta_split.pdf"),
        dpi=600,
        bbox_inches="tight",
    )

    fig = []

    all_df = pd.read_csv(
        f"{config['data directory']}/all_active_phase_data.csv", sep="\t"
    )
    all_df["previous_side"] = all_df.selected_side_map.shift(1)
    all_df["previous_win"] = all_df.delta_wealth.shift(1) > 0
    all_df["stay"] = all_df.previous_side == all_df.selected_side_map
    clean_df = all_df.query("not (previous_side.isna() or selected_side_map.isna())")
    stay_prob = (
        clean_df.groupby(["previous_win", "participant_id", "eta"])["stay"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(
        1, 1, figsize=(fig_size[0] * 1.5, fig_size[1] * 1.5), sharex=True, sharey=True
    )

    sns.stripplot(
        data=stay_prob,
        x="previous_win",
        y="stay",
        hue="eta",
        dodge=True,
        ax=axes,
        label=None,
    )
    sns.boxplot(data=stay_prob, x="previous_win", y="stay", hue="eta", ax=axes)

    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(["Additive", "Multiplicative"])
    axes.set(
        ylabel="P(stay)",
        xlabel="Previous trial outcome",
        xticklabels=["no reward", "reward"],
    )

    fig.savefig(
        os.path.join(config["figure directory"], "Rev3_stay_win_lose_shift.pdf"),
        dpi=600,
        bbox_inches="tight",
    )
    fig = []
    t1 = pg.ttest(stay_prob.query("eta == 0.0 and previous_win==0")["stay"], 0.5)
    t1.insert(0, "Hypothesis", "P(stay | eta = 0, previous win = 0) != 0.5")
    t2 = pg.ttest(stay_prob.query("eta == 1.0 and previous_win==0")["stay"], 0.5)
    t2.insert(0, "Hypothesis", "P(stay | eta = 1, previous win = 0) != 0.5")
    t3 = pg.ttest(stay_prob.query("eta == 0.0 and previous_win==1")["stay"], 0.5)
    t3.insert(0, "Hypothesis", "P(stay | eta = 0, previous win = 1) != 0.5")
    t4 = pg.ttest(stay_prob.query("eta == 1.0 and previous_win==1")["stay"], 0.5)
    t4.insert(0, "Hypothesis", "P(stay | eta = 1, previous win = 1) != 0.5")

    results = pd.concat([t1, t2, t3, t4])
    results["BF01"] = 1 / results["BF10"].astype(float)
    results["Test"] = "win stay / lose shift"

    reg_data = pd.read_csv("r_analyses/full_data_regression.tsv", sep="\t")
    reg_data["delta"] = (
        reg_data["1.0_partial_pooling"] - reg_data["0.0_partial_pooling"]
    )
    reg_data["eta_mult_z"] = (
        reg_data["1.0_partial_pooling"] - reg_data["1.0_partial_pooling"].mean()
    ) / reg_data["1.0_partial_pooling"].std(ddof=0)
    reg_data["eta_add_z"] = (
        reg_data["0.0_partial_pooling"] - reg_data["0.0_partial_pooling"].mean()
    ) / reg_data["0.0_partial_pooling"].std(ddof=0)
    reg_data["delta_z"] = (reg_data["delta"] - reg_data["delta"].mean()) / reg_data[
        "delta"
    ].std(ddof=0)

    fig, axes = plt.subplots(
        1, 2, figsize=(fig_size[0] * 2.5, fig_size[1] * 1.5), sharex=True, sharey=True
    )
    sns.regplot(data=reg_data, x="eta_add_z", y="delta_z", ax=axes[0])
    sns.regplot(data=reg_data, x="eta_mult_z", y="delta_z", ax=axes[1])
    plt.tight_layout()
    axes[0].set(ylabel=r"$z(\eta_{mult} - \eta_{add})$", xlabel=r"$z(\eta_{add})$")
    axes[1].set(ylabel=r"$z(\eta_{mult} - \eta_{add})$", xlabel=r"$z(\eta_{mult})$")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["top", "right"]].set_visible(False)

    cor1 = pg.pairwise_corr(data=reg_data, columns=["delta_z", "eta_add_z"])
    cor2 = pg.pairwise_corr(data=reg_data, columns=["delta_z", "eta_mult_z"])
    ranking = pd.concat([cor1, cor2])
    ranking["Test"] = "correlation delta(z)"

    fig.savefig(
        os.path.join(config["figure directory"], "Rev4_correlation_delta_z.pdf"),
        dpi=600,
        bbox_inches="tight",
    )

    pd.concat([pg_test, results, ranking]).to_csv(
        os.path.join(config["figure directory"], "Rev_statistics.tsv")
    )

    print(f"\n--- Code ran in {(time.time() - start_time):.2f} seconds ---")


if __name__ == "__main__":
    import sys
    from codebase.utils import write_provenance

    command = "\t".join(sys.argv)
    print(sys.argv)
    write_provenance(command)
    try:
        main()
        write_provenance("executed successfully")
    except Exception as e:
        print(e)
        traceback.print_exc()
        write_provenance("FAILED!!")
