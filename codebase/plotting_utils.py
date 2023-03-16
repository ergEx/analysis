import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm

from .utils import logistic_regression, read_Bayesian_output


def read_relevant_files(path):

    passive_output_file = os.path.join(path, "all_passive_phase_data.csv")
    if os.path.isfile(passive_output_file):
        passive_phase_df = pd.read_csv(os.path.join(path, "all_passive_phase_data.csv"), sep="\t")
    else:
        passive_phase_df = None

    indifference_eta_estimation_output_file = os.path.join(path, "all_active_phase_data.csv")
    if os.path.isfile(indifference_eta_estimation_output_file):
        indifference_eta_df = pd.read_csv(
            os.path.join(path, "all_active_phase_data.csv"), sep="\t"
        )
    else:
        ValueError("All data not saved!")

    bayesian_parameter_estimation_output_file = os.path.join(
        path, "Bayesian_parameter_estimation.mat"
    )
    if os.path.isfile(bayesian_parameter_estimation_output_file):
        bayesian_samples_parameter_estimation = read_Bayesian_output(
            bayesian_parameter_estimation_output_file
        )
    else:
        bayesian_samples_parameter_estimation = None
        print("Bayesian parameter estimation output not found!")

    bayesian_model_selection_output_file = os.path.join(path, "model_selection.mat")
    if os.path.isfile(bayesian_model_selection_output_file):
        bayesian_samples_model_selection = read_Bayesian_output(
            bayesian_parameter_estimation_output_file
        )
    else:
        bayesian_samples_model_selection = None
        print("Bayesian model selection output not found!")

    return (
        passive_phase_df,
        indifference_eta_df,
        bayesian_samples_parameter_estimation,
        bayesian_samples_model_selection,
    )


def plot_passive_trajectory(
    df: pd.DataFrame, n_passive_runs: int, reset: int, c: int, ax: plt.axes
):
    ax.plot(df.trial, df.wealth, color="grey")

    ax.set(xlabel="Trial, t", ylabel="Wealth, x")

    if c == 1.0:
        ax.set(yscale="log", ylabel="log Wealth (log)")

    for reset_idx in range(1, n_passive_runs):
        ax.axvline(x=reset * reset_idx, color="grey", linestyle="--")
    return ax


def passive_trajectory_special_labels(ax: plt.axes):
    ax.plot([], label="Reset", color="grey", linestyle="--")
    ax.axhline(y=1000, linestyle="--", color="black", label="Starting Wealth")
    ax.legend(loc="upper left", fontsize="xx-small")
    return ax


def plot_active_trajectory(df: pd.DataFrame, c: int, ax: plt.axes):
    ax.plot(df.trial, df.wealth, color="grey")

    ax.set(xlabel="Trial, t", ylabel="Wealth, x")

    if c == 1.0:
        ax.set(yscale="log", ylabel="log Wealth (log)")

    return ax


def active_trajectory_special_labels(limits: dict, ax: plt.axes):
    ax.axhline(
        y=limits[0], linestyle="--", linewidth=1, color="red", label="Upper Bound",
    )
    ax.axhline(y=1000, linestyle="--", color="black", label="Starting Wealth")

    ax.axhline(
        y=limits[1], linestyle="--", linewidth=1, color="red", label="Lower Bound",
    )
    ax.legend(loc="upper left", fontsize="xx-small")

    return ax


def plot_indifference_eta(df: pd.DataFrame, pal: sns.palettes, ax: plt.axes):
    df["tmp"] = 1
    pt.RainCloud(
        x="tmp",
        hue="min_max_sign",
        y="indif_eta",
        data=df,
        ax=ax,
        bw=0.3,
        orient="h",
        palette=pal,
        alpha=0.5,
        dodge=True,
    )
    ax.legend().set_visible(False)
    ax.set(xlabel="Indifference eta", ylabel="", yticklabels=[" "])
    return ax
    # plot_specs = {"color": {0: "orange", 1: "b"}, "sign": {0: ">", 1: "<"}}
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # for ii, choice in enumerate(df["selected_side_map"]):
    #    trial = df.loc[ii, :]
    #    if np.isnan(trial.indif_eta):
    #        continue
    #    ax.plot(
    #        trial.indif_eta,
    #        ii,
    #        marker=plot_specs["sign"][trial.min_max_sign],
    #        color=plot_specs["color"][trial.min_max_color],
    #    )

    # ax.set(title=f"Indifference eta", xlabel="Riskaversion ($\eta$)")
    # ax.axes.yaxis.set_visible(False)
    # ax.plot([], marker="<", color="b", label="Upper bound")
    # ax.plot([], marker=">", color="orange", label="Lower bound")

    # ax.legend(loc="upper left", fontsize="xx-small")

    # fig.savefig(os.path.join(save_path, f"{save_str}.png"))
    # plt.close(fig)


def plot_choice_probabilities(df: pd.DataFrame, ax: plt.axes):
    bins = [-np.inf, -0.5, 0, 1.0, 1.5, np.inf]
    min_df = df[df["min_max_sign"] == 0]
    max_df = df[df["min_max_sign"] == 1]
    min_count, _ = np.histogram(min_df["indif_eta"], bins=bins)
    max_count, _ = np.histogram(max_df["indif_eta"], bins=bins)
    choice_probs = [max_count[i] / (max_count[i] + min_count[i]) for i in range(len(min_count))]
    ticks = ["<-0.5", "-1 - 0", "0 - 1", "1 - 1.5", ">1.5"]

    ax.bar(ticks, choice_probs)
    ax.set(ylim=[0, 1], yticks=np.linspace(0, 1, 11))
    ax.tick_params(axis="x", labelrotation=45)

    return ax


def plot_indif_eta_logistic_reg(df: pd.DataFrame, ax: plt.axes):
    # Indifference eta logistic regression
    df_tmp_1 = df[df["min_max_val"] == 1]
    df_tmp_0 = df[df["min_max_val"] == 0]

    (
        x_test,
        pred,
        lower,
        upper,
        decision_boundary,
        decision_boundary_lower,
        decision_boundary_upper,
        _,
    ) = logistic_regression(df)

    ax.fill_between(
        x_test,
        lower,
        upper,
        where=upper >= lower,
        facecolor="grey",
        interpolate=True,
        alpha=0.5,
        label="95 % CI",
    )

    ax.plot(x_test, pred, color="black")

    sns.regplot(
        x=np.array(df_tmp_1.indif_eta),
        y=np.array(df_tmp_1.min_max_val),
        fit_reg=False,
        y_jitter=0.05,
        ax=ax,
        label="Upper Bound",
        scatter_kws={"alpha": 0.5, "s": 3},
    )
    sns.regplot(
        x=np.array(df_tmp_0.indif_eta),
        y=np.array(df_tmp_0.min_max_val),
        fit_reg=False,
        y_jitter=0.05,
        ax=ax,
        label="Lower Bound",
        scatter_kws={"alpha": 0.5, "s": 3},
    )

    ax.axhline(y=0.5, color="grey", linestyle="--")

    ax.set(
        ylabel="",
        xlabel="Indifference eta",
        yticks=[0, 0.5, 1],
        ylim=(-0.25, 1.25),
        xticks=np.linspace(-5, 5, 11),
        xlim=[-5, 5],
    )
    ax.axvline(x=decision_boundary, color="red", linestyle="--", ymax=0.5, label="Best estimate")
    ax.legend(loc="upper left", fontsize="xx-small")

    return ax


def plot_bayesian_estimation(dist: np.array, ax: plt.axes):
    sns.kdeplot(dist, ax=ax)
    xs, ys = ax.lines[-1].get_data()
    ax.fill_between(xs, ys, color="red", alpha=0.05)
    mode_idx = np.argmax(ys)
    ax.vlines(xs[mode_idx], 0, ys[mode_idx], ls="--", color="red", label="Prediction")
    ax.set(
        ylabel="", xticks=np.linspace(-1, 2, 7), xlim=[-1, 2], xlabel="Risk aversion estimate",
    )

    ax.legend(loc="upper left", fontsize="xx-small")

    return ax


def plot_bayesian_model_selection(dist: np.array, ax: plt.axis, n_subjects):
    pass
    sns.heatmap(dist, square=False, ax=ax, cmap="binary")
    ax.set(
        title="Model Selection",
        yticklabels=[str(x + 1) for x in list(range(n_subjects))],
        xticklabels=["Dynamic invariant", "Dynamic specific"],
    )


def plot_parameter_estimation_subject_wise(
    save_path: str,
    data_variant: str,
    subjects,
    n_agents: int,
    condition_specs,
    passive_phase_df: pd.DataFrame,
    active_phase_df: pd.DataFrame,
    bayesian_samples: np.array,
    pal,
    n_passive_runs: int = 3,
    reset: int = 45,
):
    fig_passive_all, ax_passive_all = plt.subplots(1, 2, figsize=(10, 5))
    fig_active_all, ax_active_all = plt.subplots(1, 2, figsize=(10, 5))

    for c in range(2):
        ax_passive_all[c] = passive_trajectory_special_labels(ax=ax_passive_all[c])

        ax_active_all[c] = active_trajectory_special_labels(
            limits=condition_specs["active_limits"][c], ax=ax_active_all[c]
        )

    for i, subject1 in enumerate(subjects):
        for j in range(n_agents):
            subject = f"{j}_{subject1}" if data_variant == "0_simulation" else subject1
            print(f"\nSubject {subject}")

            fig_passive, ax_passive = plt.subplots(1, 2, figsize=(10, 5))
            fig_active, ax_active = plt.subplots(1, 2, figsize=(10, 5))
            fig_indif_eta, ax_indif_eta = plt.subplots(1, 2, figsize=(10, 5))
            fig_choice_prob, ax_choice_prob = plt.subplots(1, 2, figsize=(10, 5))
            fig_log_reg, ax_log_reg = plt.subplots(1, 2, figsize=(10, 5))
            fig_bayesian, ax_bayesian = plt.subplots(1, 1, figsize=(10, 5))

            for c, condition in enumerate(condition_specs["lambd"]):
                print(f"Condition {condition}")

                """PASIVE PHASE"""
                if data_variant != "0_simulation":
                    passive_subject_df = passive_phase_df.query(
                        "participant_id == @subject1 and eta == @condition"
                    ).reset_index(drop=True)

                    ax_passive[c] = plot_passive_trajectory(
                        df=passive_subject_df,
                        n_passive_runs=n_passive_runs,
                        reset=reset,
                        c=c,
                        ax=ax_passive[c],
                    )

                    ax_passive[c] = passive_trajectory_special_labels(ax=ax_passive[c])

                    ax_passive_all[c] = plot_passive_trajectory(
                        df=passive_subject_df,
                        n_passive_runs=n_passive_runs,
                        reset=reset,
                        c=c,
                        ax=ax_passive_all[c],
                    )

                """ACTIVE PHASE"""
                if data_variant == "0_simulation":
                    active_subject_df = active_phase_df.query(
                        "agent == @subject and eta == @condition and indif_eta.notnull()",
                        engine="python",
                    ).reset_index(drop=True)
                else:
                    active_subject_df = active_phase_df.query(
                        "no_response != True and participant_id == @subject and eta == @condition and indif_eta.notnull()",
                        engine="python",
                    ).reset_index(drop=True)

                ax_active[c] = plot_active_trajectory(df=active_subject_df, c=c, ax=ax_active[c],)
                ax_active[c] = active_trajectory_special_labels(
                    limits=condition_specs["active_limits"][c], ax=ax_active[c]
                )

                ax_active_all[c] = plot_active_trajectory(
                    df=active_subject_df, c=c, ax=ax_active_all[c],
                )

                ax_indif_eta[c] = plot_indifference_eta(
                    df=active_subject_df, pal=pal, ax=ax_indif_eta[c],
                )

                ax_choice_prob[c] = plot_choice_probabilities(
                    df=active_subject_df, ax=ax_choice_prob[c]
                )

                try:
                    ax_log_reg[c] = plot_indif_eta_logistic_reg(
                        df=active_subject_df, ax=ax_log_reg[c]
                    )
                except:
                    pass

                if bayesian_samples is not None:
                    eta_dist = bayesian_samples["eta"][:, :, n_agents * i + j, c].flatten()
                    ax_bayesian = plot_bayesian_estimation(eta_dist, ax=ax_bayesian)

            fig_passive.savefig(
                os.path.join(save_path, f"1_1_passive_trajectory_{subject1}_{j}.png")
            )
            fig_active.savefig(
                os.path.join(save_path, f"1_2_active_trajectory_{subject1}_{j}.png")
            )
            fig_indif_eta.savefig(os.path.join(save_path, f"1_3_indif_eta_{subject1}_{j}.png"))
            fig_choice_prob.savefig(os.path.join(save_path, f"1_4_choice_prob_{subject1}_{j}.png"))
            fig_log_reg.savefig(os.path.join(save_path, f"1_5_log_reg_{subject1}_{j}.png"))
            fig_bayesian.savefig(os.path.join(save_path, f"1_6_bayesian_{subject1}_{j}.png"))
    fig_passive_all.savefig(os.path.join(save_path, f"0_1_passive_trajectories"))
    fig_active_all.savefig(os.path.join(save_path, f"0_2_active_trajectories"))
    plt.close("all")


def plot_parameter_estimation_all_data_as_one(
    save_path: str,
    data_variant: str,
    condition_specs: dict,
    df: pd.DataFrame,
    bayesian_samples: np.array,
    pal,
):

    fig_indif_eta, ax_indif_eta = plt.subplots(1, 2, figsize=(10, 5))
    fig_log_reg, ax_log_reg = plt.subplots(1, 2, figsize=(10, 5))
    fig_bayesian, ax_bayesian = plt.subplots(1, 2, figsize=(10, 5))
    for c, condition in enumerate(condition_specs["lambd"]):
        print(f'Condition {c+1} of {len(condition_specs["lambd"])}')
        if data_variant == "0_simulation":
            df_c = df.query(
                "eta == @condition and indif_eta.notnull()", engine="python"
            ).reset_index(drop=True)
        else:
            df_c = df.query(
                "no_response != True and eta == @condition and indif_eta.notnull()",
                engine="python",
            ).reset_index(drop=True)

        ax_indif_eta[c] = plot_indifference_eta(df=df_c, pal=pal, ax=ax_indif_eta[c])

        ax_log_reg[c] = plot_indif_eta_logistic_reg(df=df_c, ax=ax_log_reg[c])

        if bayesian_samples is not None:
            eta_dist = bayesian_samples["mu_eta"][:, :, c].flatten()
            ax_bayesian[c] = plot_bayesian_estimation(dist=eta_dist, ax=ax_bayesian[c])
    fig_indif_eta.tight_layout()
    fig_indif_eta.savefig(os.path.join(save_path, f"0_3_indif_eta.png"))
    fig_log_reg.savefig(os.path.join(save_path, f"0_4_log_reg.png"))
    fig_bayesian.savefig(os.path.join(save_path, f"0_5_bayesian.png"))


def plot_bayesian_model_selection_subject_wise(save_path: str, subjects, samples: np.array):
    print("model selection not implemented yet")
    return  # NOT IMPLEMENTED YET
    dist = np.empty([2, len(subjects)])
    for i, subject in enumerate(subjects):
        count = Counter(samples[:, :, i].flatten())
        dist[0, i] = sum(filter(None, [count.get(key) for key in [1, 3, 5, 7]]))
        dist[1, i] = sum(filter(None, [count.get(key) for key in [2, 4, 6, 8]]))

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_bayesian_model_selection(dist, ax, len(subjects))
    fig.tight_layout()
    fig.savefig(
        os.path.join(save_path, f"active_results_bayesian_model_selection_subject_wise.png")
    )


def plot_bayesian_model_selection_all_as_one(save_path: str, samples: np.array):
    print("model selection not implemented yet")
    return  # NOT IMPLEMENTED YET
    dist = np.empty(2)
    count = Counter(samples[:, :, :].flatten())
    dist[0] = sum(filter(None, [count.get(key) for key in [1, 3, 5, 7]]))
    dist[1] = sum(filter(None, [count.get(key) for key in [2, 4, 6, 8]]))

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_bayesian_model_selection(np.reshape(dist, [1, 2]), ax, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"active_results_bayesian_model_selection_aggregated.png"))


def generate_sim_overview_data(
    save_path: str,
    df: pd.DataFrame,
    subjects,
    n_agents: int,
    condition_specs: dict,
    bayesian_samples: np.array,
    n_samples_log_reg: int = 100,
):
    N = n_agents * len(subjects)
    idx_log_reg = 0
    idx_bayesian = 0
    if bayesian_samples is not None:
        n_samples_bayesian = (
            np.shape(bayesian_samples["eta"])[0] * np.shape(bayesian_samples["eta"])[1]
        )
    else:
        n_samples_bayesian = 1
    data_best_fit = {
        "log_reg": {"0.0": [None] * N, "1.0": [None] * N, "kind": [None] * N},
        "bayesian": {"0.0": [None] * N, "1.0": [None] * N, "kind": [None] * N},
    }

    data_confidence = {
        "log_reg": {
            "0.0": [None] * N * n_samples_log_reg,
            "1.0": [None] * N * n_samples_log_reg,
            "kind": [None] * N * n_samples_log_reg,
        },
        "bayesian": {
            "0.0": [None] * N * n_samples_bayesian,
            "1.0": [None] * N * n_samples_bayesian,
            "kind": [None] * N * n_samples_bayesian,
        },
    }

    for i, subject1 in enumerate(subjects):
        print(subject1)
        for j in range(n_agents):
            subject = f"{j}_{subject1}"
            data_best_fit["log_reg"][f"kind"][n_agents * i + j] = subject1
            data_best_fit["bayesian"][f"kind"][n_agents * i + j] = subject1
            data_confidence["log_reg"][f"kind"][idx_log_reg : idx_log_reg + n_samples_log_reg] = [
                subject1
            ] * n_samples_log_reg
            data_confidence["bayesian"][f"kind"][
                idx_bayesian : idx_bayesian + n_samples_bayesian
            ] = [subject1] * n_samples_bayesian
            for c, condition in enumerate(condition_specs["lambd"]):
                # Logistic regression
                df_tmp = df.query(
                    "agent == @subject and eta == @condition and indif_eta.notnull()",
                    engine="python",
                ).reset_index(drop=True)
                try:
                    (_, _, _, _, decision_boundary, _, _, std_dev,) = logistic_regression(df_tmp)
                    data_best_fit["log_reg"][f"{c}.0"][n_agents * i + j] = decision_boundary
                    data_confidence["log_reg"][f"{c}.0"][
                        idx_log_reg : idx_log_reg + n_samples_log_reg
                    ] = np.random.normal(decision_boundary, std_dev, n_samples_log_reg)
                except Exception as e:
                    pass

                # Bayesian
                try:
                    eta_dist = bayesian_samples["eta"][:, :, n_agents * i + j, c].flatten()
                    kde = sm.nonparametric.KDEUnivariate(eta_dist).fit()
                    data_best_fit["bayesian"][f"{c}.0"][n_agents * i + j] = kde.support[
                        np.argmax(kde.density)
                    ]
                    data_confidence["bayesian"][f"{c}.0"][
                        idx_bayesian : idx_bayesian + n_samples_bayesian
                    ] = eta_dist

                except Exception as e:
                    pass
            idx_log_reg += n_samples_log_reg
            idx_bayesian += n_samples_bayesian
    b_log_reg = pd.DataFrame.from_dict(data_best_fit["log_reg"])
    c_log_reg = pd.DataFrame.from_dict(data_confidence["log_reg"])
    b_log_reg.to_csv(os.path.join(save_path, "b_log_reg.csv"), sep="\t")
    c_log_reg.to_csv(os.path.join(save_path, "c_log_reg.csv"), sep="\t")

    b_bayesian = pd.DataFrame.from_dict(data_best_fit["bayesian"])
    c_bayesian = pd.DataFrame.from_dict(data_confidence["bayesian"])
    b_bayesian.to_csv(os.path.join(save_path, "b_bayesian.csv"), sep="\t")
    c_bayesian.to_csv(os.path.join(save_path, "c_bayesian.csv"), sep="\t")

    return b_log_reg, c_log_reg, b_bayesian, c_bayesian


def plot_simulation_overview(
    save_path, subjects, n_agents, b_log_reg, c_log_reg, b_bayesian, c_bayesian
):
    print("PLOTTING BEST FIT")
    if n_agents > 3:
        try:
            b_log_reg = b_log_reg[b_log_reg["kind"] != "random"]
            log_best_plot = sns.jointplot(
                data=b_log_reg,
                x="0.0",
                y="1.0",
                fill=True,
                hue="kind",
                bw_method=0.8,
                legend=False,
                alpha=0.5,
                height=10,
                kind="kde",
                xlim=[-0.6, 2.5],
                ylim=[-0.6, 2.5],
                marginal_kws={"common_norm": False},
            )
        except:
            log_best_plot = plt.figure()
            pass
        try:
            b_bayesian = b_bayesian[b_bayesian["kind"] != "random"]
            bayesian_best_plot = sns.jointplot(
                data=b_bayesian,
                x="0.0",
                y="1.0",
                fill=True,
                hue="kind",
                bw_method=0.8,
                legend=False,
                alpha=0.5,
                height=10,
                kind="kde",
                xlim=[-0.6, 2.5],
                ylim=[-0.6, 2.5],
                marginal_kws={"common_norm": False},
            )
        except:
            bayesian_best_plot = plt.figure()
            pass
    else:
        try:
            b_log_reg = b_log_reg[b_log_reg["kind"] != "random"]
            log_best_plot = sns.jointplot(
                data=b_log_reg,
                x="0.0",
                y="1.0",
                hue="kind",
                legend=False,
                alpha=0.7,
                height=10,
                kind="scatter",
                xlim=[-0.6, 2.5],
                ylim=[-0.6, 2.5],
                marginal_kws={"common_norm": False},
            )
        except:
            log_best_plot = plt.figure()
            pass
        try:
            b_bayesian = b_bayesian[b_bayesian["kind"] != "random"]
            bayesian_best_plot = sns.jointplot(
                data=b_bayesian,
                x="0.0",
                y="1.0",
                hue="kind",
                legend=False,
                alpha=0.7,
                height=10,
                kind="scatter",
                xlim=[-0.6, 2.5],
                ylim=[-0.6, 2.5],
                marginal_kws={"common_norm": False},
            )
        except:
            bayesian_best_plot = plt.figure()
            pass

    log_best_plot.savefig(os.path.join(save_path, "simulation_overview_log_reg_best_estimate.png"))
    bayesian_best_plot.savefig(
        os.path.join(save_path, "simulation_overview_bayesian_best_estimate.png")
    )

    print("PLOTTING W. CONFIDENCE INTERVAL")
    try:
        c_log_reg = c_log_reg[c_log_reg["kind"] != "random"]
        log_confidence_plot = sns.jointplot(
            data=c_log_reg,
            x="0.0",
            y="1.0",
            fill=True,
            hue="kind",
            bw_method=0.8,
            legend=False,
            alpha=0.5,
            height=10,
            kind="kde",
            xlim=[-0.6, 2.5],
            ylim=[-0.6, 2.5],
            marginal_kws={"common_norm": False},
        )
    except:
        log_confidence_plot = plt.figure()
        pass

    try:
        c_bayesian = c_bayesian[c_bayesian["kind"] != "random"]
        bayesian_confidence_plot = sns.jointplot(
            data=c_bayesian,
            x="0.0",
            y="1.0",
            fill=True,
            hue="kind",
            bw_method=0.8,
            legend=False,
            alpha=0.5,
            height=10,
            kind="kde",
            xlim=[-0.6, 2.5],
            ylim=[-0.6, 2.5],
            marginal_kws={"common_norm": False},
        )
    except:
        bayesian_confidence_plot = plt.figure()
        pass

    log_confidence_plot.savefig(
        os.path.join(save_path, f"simulation_overview_log_reg_incl_uncertainty.png")
    )
    bayesian_confidence_plot.savefig(
        os.path.join(save_path, f"simulation_overview_bayesian_incl_uncertainty.png")
    )

