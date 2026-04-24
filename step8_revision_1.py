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

    x1_left = all_df['wealth'] + all_df['x1_1']
    x2_left = all_df['wealth'] + all_df['x1_2']
    x1_right = all_df['wealth'] + all_df['x2_1']
    x2_right = all_df['wealth'] + all_df['x2_2']

    left_mean = (x1_left + x2_left) / 2
    right_mean = (x1_right + x2_right) / 2

    left_var = ((x1_left - left_mean) ** 2 + (x2_left - left_mean) ** 2) / 2
    right_var = ((x1_right - right_mean) ** 2 + (x2_right - right_mean) ** 2) / 2

    selected_var = all_df.selected_side_map * left_var + (1 - all_df.selected_side_map) * right_var
    unselected_var = (1 - all_df.selected_side_map) * left_var  + all_df.selected_side_map * right_var

    selected_mean = all_df.selected_side_map * left_mean + (1 - all_df.selected_side_map) * right_mean
    unselected_mean = (1 - all_df.selected_side_map) * left_mean  + all_df.selected_side_map * right_mean


    all_df["previous_side"] = all_df.groupby(["participant_id", "eta"]).selected_side_map.shift(1)
    all_df["previous_win"] = all_df.groupby(["participant_id", "eta"]).delta_wealth.shift(1) > 0
    all_df["stay"] = all_df.previous_side == all_df.selected_side_map

    all_df["risky_choice"] = selected_var.values > unselected_var.values
    all_df["previous_risky"] = all_df.groupby(["participant_id", "eta"]).risky_choice.shift(1)
    all_df["risk_stay"] = all_df["risky_choice"] == all_df["previous_risky"]

    all_df["better_mean"] = selected_mean.values > unselected_mean.values
    all_df["previous_mean"] = all_df.groupby(["participant_id", "eta"]).better_mean.shift(1)
    all_df["mean_stay"] = all_df["better_mean"] == all_df["previous_mean"]

    clean_df = all_df.query("not (previous_side.isna() or selected_side_map.isna())")

    stay_prob_side = (
        clean_df.groupby(["previous_win", "participant_id", "eta"])["stay"]
        .mean()
        .reset_index()
    )

    stay_prob_var = (
        clean_df.groupby(["previous_win", "participant_id", "eta"])["risk_stay"]
        .mean()
        .reset_index()
    )

    stay_prob_risky = (
        clean_df.query("previous_risky == 1").groupby(["previous_win", "participant_id", "eta"])["risk_stay"]
        .mean()
        .reset_index()
    )

    def logit(p):
        return np.log(p / (1 - p))

    def do_ttest(stay_df, outcome='stay', eta = 0):

        df = pg.ttest(logit(stay_df.query("previous_win==1 and eta == @eta")[outcome].values + np.finfo(float).eps),
                logit(stay_df.query("previous_win==0 and eta == @eta")[outcome].values + np.finfo(float).eps),
                paired=True)

        delta = (logit(stay_df.query("previous_win==1 and eta == @eta")[outcome].values + np.finfo(float).eps)  -
                logit(stay_df.query("previous_win==0 and eta == @eta")[outcome].values + np.finfo(float).eps))

        return df, delta

    t1, delta1 = do_ttest(stay_prob_side, eta=0)
    t1.insert(0, "Hypothesis", "P(stay side | eta = 0, previous win = 1) != P(stay side | eta = 0, previous win = 0)")
    t2, delta2 = do_ttest(stay_prob_side, eta=1)
    t2.insert(0, "Hypothesis", "P(stay | eta = 1, previous win = 1) != P(stay | eta = 1, previous win = 0)")

    t3, delta3 = do_ttest(stay_prob_var, eta=0, outcome="risk_stay")
    t3.insert(0, "Hypothesis", "P(stay variance | eta = 0, previous win = 1) != P(stay variance | eta = 0, previous win = 0)")
    t4, delta4 = do_ttest(stay_prob_var, eta=1, outcome="risk_stay")
    t4.insert(0, "Hypothesis", "P(stay variance| eta = 1, previous win = 1) != P(stay  variance| eta = 1, previous win = 0)")

    t5, delta5 = do_ttest(stay_prob_risky, eta=0, outcome="risk_stay")
    t5.insert(0, "Hypothesis", "P(stay risky | eta = 0, previous win = 1) != P(stay risky | eta = 0, previous win = 0)")
    t6, delta6 = do_ttest(stay_prob_risky, eta=1, outcome="risk_stay")
    t6.insert(0, "Hypothesis", "P(stay risky | eta = 1, previous win = 1) != P(stay risky | eta = 1, previous win = 0)")

    new_df_dict = {
        "delta": [],
        "session": [],
        "test": []}

    for d, s, t in zip([delta1, delta2, delta3, delta4, delta5, delta6], ['additive', 'multiplicative'] * 3,
                    ['Stay: Same side'] * 2 + ["Stay: Same variance"] * 2 + ["Stay: Higher variance"] * 2):
        new_df_dict["delta"].extend(d.tolist())
        new_df_dict["session"].extend([s] * len(d.tolist()))
        new_df_dict["test"].extend([t] * len(d.tolist()))

    new_df_dict = pd.DataFrame(new_df_dict)


    fig = plt.figure(
        figsize=(fig_size[0] * 1.5 * 3, fig_size[1] * 1.5)
    )

    ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((1, 4), (0, 3))

    axes = [ax1, ax2]

    sns.stripplot(
        data=new_df_dict,
        x="test",
        y="delta",
        hue="session",
        dodge=True,
        ax=axes[0],
        label=None,
    )

    axes[0].set(ylim=[-2, 2])

    x_coords = axes[0].get_xticks()
    x_coords_err = np.array([[x1 - 0.3, x1 + 0.3] for x1 in x_coords]).ravel()

    for d, t, xc in zip([delta1, delta2, delta3, delta4, delta5, delta6], [t1, t2, t3, t4, t5, t6], x_coords_err):
        axes[0].scatter(xc, d.mean(), color="black")
        axes[0].errorbar(xc, d.mean(), yerr=np.abs(t["CI95%"].values[0][:, None] - d.mean()), color="black", capsize=3)

    axes[0].axhline(0, color="black", alpha=0.5, linestyle="--")

    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set(
        xlabel="",
        ylabel="Delta logit(p(stay))"
    )

    sns.stripplot(
        data=new_df_dict.query("test == 'Stay: Higher variance'"),
        x="test",
        y="delta",
        hue="session",
        dodge=True,
        ax=axes[1],
        label=None,
    )

    x_coords = axes[1].get_xticks()
    x_coords_err = np.array([[x1 - 0.3, x1 + 0.3] for x1 in x_coords]).ravel()

    for d, t, xc in zip([delta5, delta6], [t5, t6], x_coords_err):
        axes[1].scatter(xc, d.mean(), color="black")
        axes[1].errorbar(xc, d.mean(), yerr=np.abs(t["CI95%"].values[0][:, None] - d.mean()), color="black", capsize=3)

    axes[1].axhline(0, color="black", alpha=0.5, linestyle="--")

    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set(
        xlabel="",
        ylabel="Delta logit(p(stay))"
    )


    axes[0].set(title = "Overview Delta p(stay)")
    axes[1].set(title = "Zoomed out: Delta p(stay | Higher variance)")

    plt.tight_layout()


    fig.savefig(
        os.path.join(config["figure directory"], "Rev3_rl_heuristics.pdf"),
        dpi=600,
        bbox_inches="tight",
    )

    results = pd.concat([t1, t2, t3, t4, t5, t6])
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
