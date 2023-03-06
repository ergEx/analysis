import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
import statsmodels.api as sm

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
    df: pd.DataFrame, save_path: str, save_str: str, lambd: float, n_passive_runs: int, reset: int,
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(df.trial, df.wealth, color="black")

    for reset_idx in range(1, n_passive_runs):
        ax.axvline(x=reset * reset_idx, color="grey", linestyle="--")
    ax.plot([], label="Reset", color="grey", linestyle="--")
    ax.legend(loc="upper left", fontsize="xx-small")
    ax.set(title=f"Passive wealth", xlabel="Trial", ylabel=f"Wealth")
    if lambd == 1.0:
        ax.set(yscale="log", ylabel="Wealth (log)")
    fig.savefig(os.path.join(save_path, f"{save_str}.png"))
    plt.close(fig)


def plot_active_trajectory(
    df: pd.DataFrame, save_path: str, save_str: str, active_limits: dict, c: int,
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(df.trial, df.wealth, color="black")
    ax.set(title=f"Active wealth", xlabel="Trial", ylabel="Wealth")

    ax.axhline(
        y=active_limits[c][0], linestyle="--", linewidth=1, color="red", label="Upper Bound"
    )
    ax.axhline(y=1000, linestyle="--", color="black", label="Starting Wealth")
    ax.axhline(
        y=active_limits[c][1], linestyle="--", linewidth=1, color="red", label="Lower Bound"
    )
    ax.legend(loc="upper left", fontsize="xx-small")
    if c == 1:
        ax.set(yscale="log", ylabel="Wealth (log)")

    fig.savefig(os.path.join(save_path, f"{save_str}.png"))
    plt.close(fig)


def plot_indifference_eta(
    df: pd.DataFrame, save_path: str, save_str: str, pal: sns.palettes,
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    pt.RainCloud(x="min_max_sign", y="indif_eta", data=df, ax=ax, bw=0.3, orient="h", palette=pal)
    ax.set(
        title="Indifference eta",
        xlabel="Indifference eta",
        ylabel="",
        yticklabels=(["Lower bound", "Upper bound"]),
    )
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"{save_str}.png"))
    plt.close(fig)

    plot_specs = {"color": {0: "orange", 1: "b"}, "sign": {0: ">", 1: "<"}}
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for ii, choice in enumerate(df["selected_side_map"]):
        trial = df.loc[ii, :]
        if np.isnan(trial.indif_eta):
            continue
        ax.plot(
            trial.indif_eta,
            ii,
            marker=plot_specs["sign"][trial.min_max_sign],
            color=plot_specs["color"][trial.min_max_color],
        )

    ax.set(title=f"Indifference eta", xlabel="Riskaversion ($\eta$)")
    ax.axes.yaxis.set_visible(False)
    ax.plot([], marker="<", color="b", label="Upper bound")
    ax.plot([], marker=">", color="orange", label="Lower bound")

    ax.legend(loc="upper left", fontsize="xx-small")

    fig.savefig(os.path.join(save_path, f"{save_str}.png"))
    plt.close(fig)


def plot_choice_probabilities(df: pd.DataFrame, save_path: str, save_str: str):
    bins = [-np.inf, -0.5, 0, 1.0, 1.5, np.inf]
    min_df = df[df["min_max_sign"] == 0]
    max_df = df[df["min_max_sign"] == 1]
    min_count, _ = np.histogram(min_df["indif_eta"], bins=bins)
    max_count, _ = np.histogram(max_df["indif_eta"], bins=bins)
    choice_probs = [max_count[i] / (max_count[i] + min_count[i]) for i in range(len(min_count))]
    ticks = ["<-0.5", "-1 - 0", "0 - 1", "1 - 1.5", ">1.5"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.bar(ticks, choice_probs)
    ax.set(title="Indif eta choice prob.", ylim=[0, 1], yticks=np.linspace(0, 1, 11))
    ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"{save_str}.png"))
    plt.close(fig)


def plot_indif_eta_logistic_reg(df: pd.DataFrame, save_path: str, save_str: str):
    # Indifference eta logistic regression
    df_tmp_1 = df[df["min_max_val"] == 1]
    df_tmp_0 = df[df["min_max_val"] == 0]

    x_test, pred, ymin, ymax, idx_m, idx_l, idx_h = logistic_regression(df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.fill_between(
        x_test,
        ymin,
        ymax,
        where=ymax >= ymin,
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
        color="b",
        scatter_kws={"alpha": 1, "s": 20},
    )
    sns.regplot(
        x=np.array(df_tmp_0.indif_eta),
        y=np.array(df_tmp_0.min_max_val),
        fit_reg=False,
        y_jitter=0.05,
        ax=ax,
        label="Lower Bound",
        color="orange",
        scatter_kws={"alpha": 1, "s": 20},
    )

    ax.axhline(y=0.5, color="grey", linestyle="--")

    ax.set(
        title=f"Logistic regression",
        ylabel="",
        xlabel="Indifference eta",
        yticks=[0, 0.5, 1],
        ylim=(-0.25, 1.25),
        xticks=np.linspace(-5, 5, 11),
        xlim=[-5, 5],
    )
    ax.axvline(x=x_test[idx_m], color="red", linestyle="--", label="Best estimate")
    ax.legend(loc="upper left", fontsize="xx-small")

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"{save_str}.png"))
    plt.close(fig)


def plot_bayesian_estimation(dist: np.array, save_path: str, save_str: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.kdeplot(dist, ax=ax)
    xs, ys = ax.lines[-1].get_data()
    ax.fill_between(xs, ys, color="red", alpha=0.05)
    mode_idx = np.argmax(ys)
    ax.vlines(xs[mode_idx], 0, ys[mode_idx], ls="--", color="red", label="Prediction")
    ax.set(
        title=f"Bayesian Model",
        ylabel="",
        xticks=np.linspace(-1, 2, 7),
        xlim=[-1, 2],
        xlabel="Risk aversion estimate",
    )

    ax.legend(loc="upper left", fontsize="xx-small")
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"{save_str}.png"))
    plt.close(fig)


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
    subjects: list[str],
    n_agents: int,
    condition_specs: dict,
    passive_phase_df: pd.DataFrame,
    active_phase_df: pd.DataFrame,
    bayesian_samples: np.array,
    pal,
    n_passive_runs: int = 3,
    reset: int = 45,
):

    for i, subject1 in enumerate(subjects):
        for j in range(n_agents):
            subject = f"{j}_{subject1}" if data_variant == "0_simulation" else subject1
            print(f"\nSubject {subject}")
            for c, condition in enumerate(condition_specs["lambd"]):
                print(f"Condition {condition}")

                """PASIVE PHASE"""
                if data_variant != "0_simulation":
                    passive_subject_df = passive_phase_df.query(
                        "participant_id == @subject1 and eta == @condition"
                    ).reset_index(drop=True)

                    plot_passive_trajectory(
                        passive_subject_df,
                        save_path,
                        f"passive_trajectory_{subject}_{c}",
                        c,
                        n_passive_runs,
                        reset,
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

                plot_active_trajectory(
                    active_subject_df,
                    save_path,
                    f"active_trajectory_{subject}_{c}",
                    condition_specs["active_limits"],
                    c,
                )

                plot_indifference_eta(
                    active_subject_df, save_path, f"Indifference_eta_{i}_{j}", pal
                )

                plot_choice_probabilities(
                    active_subject_df, save_path, f"choice_probabilities_{subject}_{c}"
                )

                try:
                    plot_indif_eta_logistic_reg(
                        active_subject_df, save_path, f"logistic_reg_{subject}_{c}"
                    )
                except:
                    pass

                if bayesian_samples is not None:
                    eta_dist = bayesian_samples["eta"][:, :, n_agents * i + j, c].flatten()
                    plot_bayesian_estimation(
                        eta_dist, save_path, f"bayesian_parameter_est_{subject}_{c}"
                    )


def plot_parameter_estimation_all_data_as_one(
    save_path: str,
    data_variant: str,
    condition_specs: dict,
    df: pd.DataFrame,
    bayesian_samples: np.array,
    pal,
):
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

        plot_indifference_eta(df_c, save_path, f"0_indifference_eta_{c}", pal)

        plot_indif_eta_logistic_reg(df_c, save_path, f"0_logistic_reg_{c}")

        if bayesian_samples is not None:
            eta_dist = bayesian_samples["mu_eta"][:, :, c].flatten()
            plot_bayesian_estimation(eta_dist, save_path, f"0_bayesian_parameter_est_{c}")


def plot_bayesian_model_selection_subject_wise(
    save_path: str, subjects: list[str], samples: np.array
):
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


def plot_simulation_overview_individuals(
    save_path: str,
    df: pd.DataFrame,
    subjects: list[dict],
    n_agents: int,
    condition_specs: dict,
    bayesian_samples: np.array,
):
    N = n_agents * len(subjects)
    data = {
        "log_reg": {"0.0": [None] * N, "1.0": [None] * N, "kind": [None] * N},
        "bayesian": {"0.0": [None] * N, "1.0": [None] * N, "kind": [None] * N},
    }

    for i, subject1 in enumerate(subjects):
        for j in range(n_agents):
            subject = f"{j}_{subject1}"
            data["log_reg"][f"kind"][n_agents * i + j] = subject1
            data["bayesian"][f"kind"][n_agents * i + j] = subject1
            for c, condition in enumerate(condition_specs["lambd"]):
                # Logistic regression
                df_tmp = df.query(
                    "agent == @subject and eta == @condition and indif_eta.notnull()",
                    engine="python",
                ).reset_index(drop=True)
                try:
                    x_test, _, _, _, idx_m, _, _ = logistic_regression(df_tmp)
                    data["log_reg"][f"{c}.0"][n_agents * i + j] = x_test[idx_m]
                except Exception as e:
                    pass

                # Bayesian
                try:
                    eta_dist = bayesian_samples["eta"][:, :, n_agents * i + j, c].flatten()
                    kde = sm.nonparametric.KDEUnivariate(eta_dist).fit()
                    data["bayesian"][f"{c}.0"][n_agents * i + j] = kde.support[
                        np.argmax(kde.density)
                    ]
                except Exception as e:
                    pass
    c_log_reg = pd.DataFrame.from_dict(data["log_reg"])
    c_log_reg = c_log_reg.dropna()
    c_bayesian = pd.DataFrame.from_dict(data["bayesian"])
    c_bayesian = c_bayesian.dropna()
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("Simulation Overview")
    ax[0].set(
        title=f"Logistic regression",
        xlabel="Additive condition",
        ylabel=f"Multiplicative condition",
    )
    ax[1].set(
        title=f"Bayesian parameter estimation",
        xlabel="Additive condition",
        ylabel=f"Multiplicative condition",
    )
    try:
        sns.kdeplot(
            data=c_log_reg,
            x="0.0",
            y="1.0",
            fill=True,
            hue="kind",
            bw_method=0.8,
            ax=ax[0],
            legend=False,
        )
    except:
        pass
    try:
        sns.kdeplot(
            data=c_bayesian,
            x="0.0",
            y="1.0",
            fill=True,
            hue="kind",
            bw_method=0.8,
            ax=ax[1],
            legend=True,
        )
    except:
        pass

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"simulation_overview_individuals.png"))


def plot_simulation_overview_group(
    save_path: str,
    df: pd.DataFrame,
    subjects: list[dict],
    condition_specs: dict,
    bayesian_samples: np.array,
):
    N = len(subjects)
    data = {
        "log_reg": {"0.0": [None] * N, "1.0": [None] * N, "kind": [None] * N},
        "bayesian": {"0.0": [None] * N, "1.0": [None] * N, "kind": [None] * N},
    }

    df["sub"] = df.agent.apply(lambda x: x[-7:])

    for i, subject1 in enumerate(subjects):
        data["log_reg"][f"kind"][i] = subject1
        data["bayesian"][f"kind"][i] = subject1
        for c, condition in enumerate(condition_specs["lambd"]):
            # Logistic regression
            df_tmp = df.query(
                "sub == @subject1 and eta == @condition and indif_eta.notnull()", engine="python",
            ).reset_index(drop=True)
            try:
                x_test, _, _, _, idx_m, _, _ = logistic_regression(df_tmp)
                data["log_reg"][f"{c}.0"][i] = x_test[idx_m]
            except Exception as e:
                pass

            # Bayesian
            try:
                eta_dist = bayesian_samples["mu_eta"][:, :, c].flatten()
                kde = sm.nonparametric.KDEUnivariate(eta_dist).fit()
                data["bayesian"][f"{c}.0"][i] = kde.support[np.argmax(kde.density)]
            except Exception as e:
                pass
    c_log_reg = pd.DataFrame.from_dict(data["log_reg"])
    c_log_reg = c_log_reg.dropna()
    c_bayesian = pd.DataFrame.from_dict(data["bayesian"])
    c_bayesian = c_bayesian.dropna()
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("Simulation Overview")
    ax[0].set(
        title=f"Logistic regression",
        xlabel="Additive condition",
        ylabel=f"Multiplicative condition",
    )
    ax[1].set(
        title=f"Bayesian parameter estimation",
        xlabel="Additive condition",
        ylabel=f"Multiplicative condition",
    )
    try:
        ax[0].scatter(c_log_reg["0.0"], c_log_reg["1.0"], label=c_log_reg["kind"])
    except:
        pass
    try:
        ax[0].scatter(c_bayesian["0.0"], c_bayesian["1.0"], label=c_bayesian["kind"])
    except:
        pass

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"simulation_overview_group.png"))
