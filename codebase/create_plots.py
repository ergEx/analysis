import os
import sys
from collections import Counter
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .experiment_specs import condition_specs, sub_specs
from .utils import logistic_regression, read_Bayesian_output


def plot_passive_trajectory(
    df: pd.DataFrame, ax: plt.axis, n_passive_runs: int, reset: int, c: int, idx: int = 0
) -> plt.axis:
    df = df.query("participant_id == @subject and eta == @condition").reset_index(drop=True)
    ax[c, idx].plot(df.trial, df.wealth)
    for reset_idx in range(1, n_passive_runs):
        ax[c, 0].axvline(x=reset * reset_idx, color="grey", linestyle="--")
    ax[c, idx].plot([], label="Reset", color="grey", linestyle="--")
    ax[c, idx].legend(loc="upper left", fontsize="xx-small")
    ax[c, idx].set(title=f"Passive wealth", xlabel="Trial", ylabel=f"Wealth")
    if c == 1:
        ax[c, idx].set_yscale("log")


def plot_active_trajectory(
    df: pd.DataFrame, ax: plt.axis, active_limits: dict, c: int, idx: int = 1
) -> plt.axis:
    ax[c, idx].plot(df.trial, df.wealth)
    ax[c, idx].set(title=f"Active wealth", xlabel="Trial", ylabel="Wealth")

    ax[c, idx].axhline(
        y=active_limits[c][0], linestyle="--", linewidth=1, color="red", label="Upper Bound"
    )
    ax[c, idx].axhline(y=1000, linestyle="--", color="black", label="Starting Wealth")
    ax[c, idx].axhline(
        y=active_limits[c][1], linestyle="--", linewidth=1, color="red", label="Lower Bound"
    )
    ax[c, idx].legend(loc="upper left", fontsize="xx-small")
    if c == 1:
        ax[c, 1].set(yscale="log", ylabel="Wealth (log)")


def plot_indifference_eta(
    df: pd.DataFrame,
    ax: plt.axis,
    plot_specs: dict,
    c: int,
    simulation_eta: float = None,
    idx: int = 2,
):
    for ii, choice in enumerate(df["selected_side_map"]):
        trial = df.loc[ii, :]
        if np.isnan(trial.indif_eta):
            continue
        ax[c, idx].plot(
            trial.indif_eta,
            ii,
            marker=plot_specs["sign"][trial.min_max_sign],
            color=plot_specs["color"][trial.min_max_color],
        )

    ax[c, idx].set(title=f"Indifference eta", xlabel="Riskaversion ($\eta$)")
    ax[c, idx].axes.yaxis.set_visible(False)
    ax[c, idx].axvline(c, linestyle="--", color="grey", label="Growth optimal")
    if simulation_eta is not None:
        ax[c, idx].axvline(simulation_eta, linestyle="--", color="green", label="Simulation eta")

    ax[c, idx].plot([], marker="<", color="b", label="Upper bound")
    ax[c, idx].plot([], marker=">", color="orange", label="Lower bound")

    ax[c, idx].legend(loc="upper left", fontsize="xx-small")


def plot_choice_probabilities(df: pd.DataFrame, ax: plt.axis, c: int, idx: int = 3):
    bins = [-np.inf, -0.5, 0, 1.0, 1.5, np.inf]
    min_df = df[df["min_max_sign"] == 0]
    max_df = df[df["min_max_sign"] == 1]
    min_count, _ = np.histogram(min_df["indif_eta"], bins=bins)
    max_count, _ = np.histogram(max_df["indif_eta"], bins=bins)
    choice_probs = [max_count[i] / (max_count[i] + min_count[i]) for i in range(len(min_count))]
    ticks = ["<-0.5", "-1 - 0", "0 - 1", "1 - 1.5", ">1.5"]

    ax[c, idx].bar(ticks, choice_probs)
    ax[c, idx].set(title="Indif eta choice prob.", ylim=[0, 1], yticks=np.linspace(0, 1, 11))
    ax[c, idx].tick_params(axis="x", labelrotation=45)


def plot_indif_eta_logistic_reg(
    df: pd.DataFrame, ax: plt.axis, c: int, simulation_eta: float = None, idx: int = 4
):
    # Indifference eta logistic regression
    df_tmp = df.query("indif_eta.notnull()", engine="python")
    df_tmp_1 = df_tmp[df_tmp["min_max_val"] == 1]
    df_tmp_0 = df_tmp[df_tmp["min_max_val"] == 0]

    print(f"Number og relevant gambles: {len(df_tmp) / len(df):.2f}")
    x_test, pred, ymin, ymax, idx_m, idx_l, idx_h = logistic_regression(df_tmp)

    ax[c, idx].fill_between(
        x_test,
        ymin,
        ymax,
        where=ymax >= ymin,
        facecolor="grey",
        interpolate=True,
        alpha=0.5,
        label="95 % CI",
    )

    ax[c, idx].plot(x_test, pred, color="black")

    sns.regplot(
        x=np.array(df_tmp_1.indif_eta),
        y=np.array(df_tmp_1.min_max_val),
        fit_reg=False,
        y_jitter=0.05,
        ax=ax[c, idx],
        label="Upper Bound",
        color="b",
        marker="<",
        scatter_kws={"alpha": 1, "s": 20},
    )
    sns.regplot(
        x=np.array(df_tmp_0.indif_eta),
        y=np.array(df_tmp_0.min_max_val),
        fit_reg=False,
        y_jitter=0.05,
        ax=ax[c, idx],
        label="Lower Bound",
        color="orange",
        marker=">",
        scatter_kws={"alpha": 1, "s": 20},
    )

    ax[c, idx].axvline(c, linestyle="--", color="grey", label="Growth optimal")
    ax[c, idx].axhline(y=0.5, color="grey", linestyle="--")

    ax[c, idx].set(
        title=f"Logistic regression",
        ylabel="",
        xlabel="Indifference eta",
        yticks=[0, 0.5, 1],
        ylim=(-0.25, 1.25),
        xticks=np.linspace(-5, 5, 11),
        xlim=[-5, 5],
    )
    if simulation_eta is not None:
        ax[c, idx].axvline(simulation_eta, linestyle="--", color="green", label="Simulation eta")
    ax[c, idx].axvline(x=x_test[idx_m], color="red", linestyle="--", label="Best estimate")
    ax[c, idx].legend(loc="upper left", fontsize="xx-small")


def plot_bayesian_estimation(
    dist: np.array, ax: plt.axis, c: int, simulation_eta: float = None, idx: int = 5
):
    sns.kdeplot(dist, ax=ax[c, idx])
    xs, ys = ax[c, idx].lines[-1].get_data()
    ax[c, idx].fill_between(xs, ys, color="red", alpha=0.05)
    mode_idx = np.argmax(ys)
    ax[c, idx].vlines(xs[mode_idx], 0, ys[mode_idx], ls="--", color="red", label="Prediction")
    ax[c, idx].axvline(c, linestyle="--", linewidth=1, color="k")
    ax[c, idx].set(
        title=f"Bayesian Model",
        ylabel="",
        xticks=np.linspace(-5, 5, 11),
        xlim=[-5, 5],
        xlabel="Risk aversion estimate",
    )

    if simulation_eta is not None:
        ax[c, idx].axvline(simulation_eta, linestyle="--", color="green", label="Simulation eta")

    ax[c, idx].legend(loc="upper left", fontsize="xx-small")


def plot_bayesian_model_selection(dist: np.array, ax: plt.axis, n_subjects):
    sns.heatmap(dist, square=False, ax=ax, cmap="binary")
    ax.set(
        title="Model Selection",
        yticklabels=[str(x + 1) for x in list(range(n_subjects))],
        xticklabels=["Dynamic invariant", "Dynamic specific"],
    )


def plot_parameter_estimation_subject_wise(
    save_path: str,
    data_variant: str,
    subjects: list[str],
    n_agents: int,
    condition_specs: dict,
    passive_phase_df: pd.DataFrame,
    n_passive_runs: int,
    reset: int,
    active_phase_df: pd.DataFrame,
    indifference_eta_plot_specs: dict,
    bayesian_samples: np.array,
):
    for i, subject1 in enumerate(subjects):

        print(f"Subject {i+1} of {len(subjects)}")
        for i in range(n_agents):
            subject = f"{i}_{subject1}"
            fig, ax = plt.subplots(2, 6, figsize=(20, 7))
            fig.suptitle(f"Subject {subject1}")
            for c, condition in enumerate(condition_specs["lambd"]):
                print(f'Condition {c+1} of {len(condition_specs["lambd"])}')

                """PASIVE PHASE"""
                if data_variant == "0_simulation":
                    ax[c, 0].plot()
                    ax[c, 0].set(title=f"Passive wealth", xlabel="Trial", ylabel=f"Wealth")
                else:
                    plot_passive_trajectory(passive_phase_df, ax, n_passive_runs, reset, c)

                """ACTIVE PHASE"""
                print(subject, condition)
                if data_variant == "0_simulation":
                    active_subject_df = active_phase_df.query(
                        "agent == @subject and eta == @condition"
                    ).reset_index(drop=True)
                    simulation_eta = float(subject1.split("x")[c])
                else:
                    active_subject_df = active_phase_df.query(
                        "no_response != True and subject_id == @subject and eta == @condition"
                    ).reset_index(drop=True)

                plot_active_trajectory(active_subject_df, ax, condition_specs["active_limits"], c)

                plot_indifference_eta(
                    active_subject_df, ax, indifference_eta_plot_specs, c, simulation_eta
                )

                plot_choice_probabilities(active_subject_df, ax, c)

                try:
                    est = plot_indif_eta_logistic_reg(active_subject_df, ax, c, simulation_eta)
                except:
                    pass

                if bayesian_samples is None:
                    ax[c, 5].plot()
                    ax[c, 5].set(
                        title=f"Bayesian Model",
                        ylabel="",
                        xticks=np.linspace(-5, 5, 11),
                        xlim=[-5, 5],
                        xlabel="Risk aversion estimate",
                    )
                else:
                    eta_dist = bayesian_samples["eta"][:, :, i, c].flatten()
                    plot_bayesian_estimation(eta_dist, ax, c, simulation_eta)

                fig.tight_layout()
                fig.savefig(os.path.join(save_path, f"Subject_{subject}.png"))


def plot_parameter_estimation_all_data_as_one(
    save_path: str,
    data_variant: str,
    condition_specs: dict,
    df: pd.DataFrame,
    indifference_eta_plot_specs: dict,
    bayesian_samples: np.array = None,
):
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"All data")
    print(condition_specs)
    for c, condition in enumerate(condition_specs["lambd"]):
        print(f'Condition {c+1} of {len(condition_specs["lambd"])}')
        if data_variant == "0_simulation":
            df_c = df.query("eta == @condition").reset_index(drop=True)
        else:
            df_c = df.query("no_response != True and eta == @condition").reset_index(drop=True)
        plot_indifference_eta(df_c, ax, indifference_eta_plot_specs, c, idx=0)

        plot_indif_eta_logistic_reg(df_c, ax, c, idx=1)

        if bayesian_samples is None:
            ax[c, 1].plot()
            ax[c, 1].set(
                title=f"Bayesian Model",
                ylabel="",
                xticks=np.linspace(-5, 5, 11),
                xlim=[-5, 5],
                xlabel="Risk aversion estimate",
            )
        else:
            eta_dist = bayesian_samples["eta"][:, :, :, c].flatten()
            plot_bayesian_estimation(eta_dist, ax, c, idx=2)

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"active_results_aggregated.png"))


def plot_bayesian_model_selection_subject_wise(
    save_path: str, subjects: list[str], samples: np.array
):
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
    dist = np.empty(2)
    count = Counter(samples[:, :, :].flatten())
    dist[0] = sum(filter(None, [count.get(key) for key in [1, 3, 5, 7]]))
    dist[1] = sum(filter(None, [count.get(key) for key in [2, 4, 6, 8]]))

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_bayesian_model_selection(np.reshape(dist, [1, 2]), ax, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"active_results_bayesian_model_selection_aggregated.png"))


if __name__ == "__main__":
    N_AGENTS = 2
    PASSIVE_RESET = 45
    N_PASSIVE_RUNS = 3
    ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
    DATA_VARIANT = "0_simulation"
    for SIMULATION_VARIANT in ["b_0_n_0"]:  # , "b_1_n_0", "b_0_n_1", "b_1_n_1"]:
        PATH = (
            os.path.join(DATA_VARIANT, SIMULATION_VARIANT)
            if DATA_VARIANT == "0_simulation"
            else DATA_VARIANT
        )

        CONDITION_SPECS = condition_specs()
        SUBJECT_SPECS = sub_specs(DATA_VARIANT)
        subjects = SUBJECT_SPECS["id"]
        INDIFFERENCE_ETA_PLOT_SPECS = {"color": {0: "orange", 1: "b"}, "sign": {0: ">", 1: "<"}}

        if DATA_VARIANT == "0_simulation":
            passive_phase_df = None
        else:
            passive_phase_df = pd.read_csv(
                os.path.join(ROOT_PATH, "data", PATH, "all_passive_phase_data.csv"), sep="\t"
            )

        active_phase_df = pd.read_csv(
            os.path.join(ROOT_PATH, "data", PATH, "all_active_phase_data.csv"), sep="\t"
        )

        save_path = os.path.join(ROOT_PATH, "figs", PATH)

        indifference_eta_estimation_output_file = os.path.join(
            ROOT_PATH, "data", PATH, "all_active_phase_data.csv"
        )
        if os.path.isfile(indifference_eta_estimation_output_file):
            ValueError("All data not saved!")

        bayesian_parameter_estimation_output_file = os.path.join(
            ROOT_PATH, "data", PATH, "Bayesian_parameter_estimation.mat"
        )
        if os.path.isfile(bayesian_parameter_estimation_output_file):
            bayesian_samples_parameter_estimation = read_Bayesian_output(
                bayesian_parameter_estimation_output_file
            )
        else:
            bayesian_samples_parameter_estimation = None
            print("Bayesian parameter estimation output not found!")

        bayesian_model_selection_output_file = os.path.join(
            ROOT_PATH, "data", PATH, "model_selection.mat"
        )
        if os.path.isfile(bayesian_model_selection_output_file):
            bayesian_samples_model_selection = read_Bayesian_output(
                bayesian_parameter_estimation_output_file
            )
        else:
            bayesian_samples_model_selection = None
            print("Bayesian model selection output not found!")

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        plot_parameter_estimation_subject_wise(
            save_path,
            DATA_VARIANT,
            subjects,
            N_AGENTS,
            CONDITION_SPECS,
            passive_phase_df,
            N_PASSIVE_RUNS,
            PASSIVE_RESET,
            active_phase_df,
            INDIFFERENCE_ETA_PLOT_SPECS,
            bayesian_samples_parameter_estimation,
        )

        plot_parameter_estimation_all_data_as_one(
            save_path,
            DATA_VARIANT,
            CONDITION_SPECS,
            active_phase_df,
            INDIFFERENCE_ETA_PLOT_SPECS,
            bayesian_samples_parameter_estimation,
        )

        if bayesian_samples_model_selection is not None:
            plot_bayesian_model_selection_subject_wise(
                save_path, subjects, bayesian_samples_model_selection
            )

            plot_bayesian_model_selection_all_as_one(save_path, bayesian_samples_model_selection)

