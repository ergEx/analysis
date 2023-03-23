import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
import yaml

from .base import get_config_filename


def plot_trajectory(ax, df, c, title_dict={0: "Additive", 1: "Multiplicative"}, session="passive"):
    ax.plot(df.trial, df.wealth)
    ax.set(title=title_dict[c], xlabel="Trial", ylabel="Wealth")

    if c == 1:
        ax.set(yscale="log", ylabel="log Wealth (log)")

    if session == "passive":
        for i in [45, 45 * 2]:
            ax.axvline(x=i, linestyle="--", color="grey")
        label = "reset" if not ax.get_legend_handles_labels()[0] else None
        ax.plot([], linestyle="--", color="grey", label=label)
        ax.legend(loc="upper left", fontsize="xx-small")

    return ax


def plot_passive(df_passive, fig_dir):
    fig_passive, ax_passive = plt.subplots(2, 1)
    for i, participant in enumerate(set(df_passive.participant_id)):
        fig_passive_subject, ax_passive_subject = plt.subplots(2, 1)
        for c, cond in enumerate(set(df_passive.eta)):
            df_tmp = df_passive.query(
                "participant_id == @participant and eta == @cond"
            ).reset_index(drop=True)

            ax_passive[c] = plot_trajectory(ax_passive[c], df_tmp, c)
            ax_passive_subject[c] = plot_trajectory(ax_passive_subject[c], df_tmp, c)

        fig_passive_subject.suptitle(f"Wealth trajectory in learning task for participant {i+1}")
        fig_passive_subject.tight_layout()
        fig_passive_subject.savefig(os.path.join(fig_dir, f"1_passive_{i}"))
    fig_passive.suptitle(f"Wealth trajectory in learning task")
    fig_passive.tight_layout()
    fig_passive.savefig(os.path.join(fig_dir, f"01_passive"))


def plot_active(df_active, fig_dir):
    fig_active, ax_active = plt.subplots(2, 1)
    for i, participant in enumerate(set(df_active.participant_id)):
        fig_active_subject, ax_active_subject = plt.subplots(2, 1)
        for c, cond in enumerate(set(df_active.eta)):
            df_tmp = df_active.query(
                "participant_id == @participant and eta == @cond"
            ).reset_index(drop=True)

            ax_active[c] = plot_trajectory(ax_active[c], df_tmp, c, session="active")
            ax_active_subject[c] = plot_trajectory(ax_active[c], df_tmp, c, session="active")

        fig_active_subject.suptitle(f"Wealth trajectories in gamblig task for participant {i+1}")
        fig_active_subject.tight_layout()
        fig_active_subject.savefig(os.path.join(fig_dir, f"2_active_{i}"))
    fig_active.suptitle("Wealth trajectories in gamblig task")
    fig_active.tight_layout()
    fig_active.savefig(os.path.join(fig_dir, f"02_active"))


def plot_raincloud(ax, df, c, pal, title_dict={0: "Additive", 1: "Multiplicative"}):
    pt.RainCloud(
        x="tmp",
        hue="min_max",
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
    ax.set(title=title_dict[c], xlabel="Indifference eta", ylabel="", yticks=[], xlim=[-6, 6])
    return ax


def plot_indif_eta(df_active, pal, fig_dir):
    fig_indif_eta, ax_indif_eta = plt.subplots(2, 1)
    for c, cond in enumerate(set(df_active.eta)):
        df_tmp = df_active.query("eta == @cond").reset_index(drop=True)

        ax_indif_eta[c] = plot_raincloud(ax_indif_eta[c], df_tmp, c, pal)

    fig_indif_eta.suptitle("Descriptive indifference eta")
    fig_indif_eta.tight_layout()
    fig_indif_eta.savefig(os.path.join(fig_dir, f"03_indif_eta"))
    for i, participant in enumerate(set(df_active.participant_id)):
        fig_indif_eta_subject, ax_indif_eta_subject = plt.subplots(2, 1)
        for c, cond in enumerate(set(df_active.eta)):
            df_tmp = df_active.query(
                "participant_id == @participant and eta == @cond"
            ).reset_index(drop=True)

            ax_indif_eta_subject[c] = plot_raincloud(ax_indif_eta_subject[c], df_tmp, c, pal)

        fig_indif_eta_subject.suptitle(f"Desctriptive indifference eta for participant {i+1}")
        fig_indif_eta_subject.tight_layout()
        fig_indif_eta_subject.savefig(os.path.join(fig_dir, f"3_indif_eta_{i}"))


def log_reg_plot(ax, df, best, c, i=0, title_dict={0: "Additive", 1: "Multiplicative"}):
    ax.fill_between(
        df.loc[:, f"x_test_{c}_{i}"],
        df.loc[:, f"confidence_lower_{c}_{i}"],
        df.loc[:, f"confidence_upper_{c}_{i}"],
        where=df.loc[:, f"confidence_upper_{c}_{i}"] >= df.loc[:, f"confidence_lower_{c}_{i}"],
        facecolor="grey",
        interpolate=True,
        alpha=0.5,
        label="95 % CI",
    )

    ax.plot(
        df.loc[:, f"x_test_{c}_{i}"], df.loc[:, f"est_{c}_{i}"], color="black",
    )

    ax.axvline(x=best, ymax=0.5, linestyle="--", color="red", label="estimate")
    ax.axhline(y=0.5, linestyle="--", color="grey")

    ax.set(
        title=title_dict[c],
        xlabel="Indifference eta",
        ylabel="",
        xlim=[-6, 6],
        yticks=[0, 0.5, 1],
        xticks=np.linspace(-5, 5, 11),
    )

    return ax


def plot_regplot(ax, df, v):
    sns.regplot(
        x=np.array(df[df.min_max == v].indif_eta),
        y=np.array(df[df.min_max == v].min_max),
        fit_reg=False,
        y_jitter=0.05,
        ax=ax,
        label="Upper Bound",
        scatter_kws={"alpha": 0.5, "s": 3},
    )

    return ax


def plot_log_reg(df_active, df_log_reg_all, df_log_reg_subject, df_best, fig_dir):

    fig_log_reg, ax_log_reg = plt.subplots(2, 1)
    for c, cond in enumerate(set(df_active.eta)):
        ax_log_reg[c] = log_reg_plot(
            ax_log_reg[c], df_log_reg_all, df_best.iloc[0][f"log_reg_best_{c}"], c
        )

        df_tmp = df_active.query("eta == @cond")

        ax_log_reg[c] = plot_regplot(ax_log_reg[c], df_tmp, 1)
        ax_log_reg[c] = plot_regplot(ax_log_reg[c], df_tmp, 0)

        ax_log_reg[c].legend(loc="upper left", fontsize="xx-small")
    fig_log_reg.suptitle("Logistic regression")
    fig_log_reg.tight_layout()
    fig_log_reg.savefig(os.path.join(fig_dir, f"04_log_reg"))

    for i, participant in enumerate(set(df_active.participant_id)):
        fig_log_reg_subject, ax_log_reg_subject = plt.subplots(2, 1)
        for c, cond in enumerate(set(df_active.eta)):

            ax_log_reg_subject[c] = log_reg_plot(
                ax_log_reg_subject[c],
                df_log_reg_subject,
                df_best.iloc[i + 1][f"log_reg_best_{c}"],
                c,
                i + 1,
            )

            df_tmp = df_active.query("participant_id == @participant and eta == @cond")

            ax_log_reg_subject[c] = plot_regplot(ax_log_reg_subject[c], df_tmp, 1)
            ax_log_reg_subject[c] = plot_regplot(ax_log_reg_subject[c], df_tmp, 0)

            ax_log_reg_subject[c].legend(loc="upper left", fontsize="xx-small")

        fig_log_reg_subject.suptitle(f"Logistic regression for participant {i+1}")
        fig_log_reg_subject.tight_layout()
        fig_log_reg_subject.savefig(os.path.join(fig_dir, f"4_log_reg_{i}"))


def plot_bayesian(df_active, df_bayesian_all, df_bayesian_subject, df_best, fig_dir):
    legend_dict = {0: "Additive", 1: "Multiplicative"}
    fig_bayesian, ax_bayesian = plt.subplots(1, 1)
    for c, cond in enumerate(set(df_active.eta)):
        ax_bayesian = sns.kdeplot(
            df_bayesian_all.loc[:, f"{c}_0"], ax=ax_bayesian, label=legend_dict[c], fill=True,
        )
        xs, ys = ax_bayesian.collections[-1].get_paths()[0].vertices.T
        ax_bayesian.fill_between(xs, ys, color="red", alpha=0.05)
        mode_idx = np.argmax(ys)
        ax_bayesian.vlines(xs[mode_idx], 0, ys[mode_idx], ls="--", color="red", label="Prediction")
    ax_bayesian.legend(loc="upper left", fontsize="xx-small")
    ax_bayesian.set(
        title="Bayesian parameter estimation",
        xlabel="Riskaversion parameter",
        ylabel="",
        yticks=[],
        xlim=[-1, 2],
    )
    fig_bayesian.savefig(os.path.join(fig_dir, "05_bayesian"))

    for i, participant in enumerate(set(df_active.participant_id)):
        fig_bayesian_subjects, ax_bayesian_subjects = plt.subplots(1, 1)
        for c, cond in enumerate(set(df_active.eta)):
            sns.kdeplot(
                df_bayesian_subject.loc[:, f"{c}_{i+1}"],
                ax=ax_bayesian_subjects,
                fill=True,
                label=legend_dict[c],
            )
            xs, ys = ax_bayesian_subjects.collections[-1].get_paths()[0].vertices.T
            ax_bayesian_subjects.fill_between(xs, ys, color="red", alpha=0.05)
            mode_idx = np.argmax(ys)
            ax_bayesian_subjects.vlines(
                xs[mode_idx], 0, ys[mode_idx], ls="--", color="red", label="Prediction"
            )

        ax_bayesian_subjects.legend(loc="upper left", fontsize="xx-small")
        ax_bayesian_subjects.set(
            title="Bayesian parameter estimation for participant {i+1}",
            xlabel="Riskaversion parameter",
            ylabel="",
            yticks=[],
            xlim=[-1, 2],
        )
        fig_bayesian_subjects.savefig(os.path.join(fig_dir, f"5_bayesian_{i}"))


def main(config_file, i, simulation_variant=""):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["plots"]["run"]:
        return

    data_dir = config["data directoty"]
    fig_dir = config["figure directoty"]

    colors = config["colors"]
    pal = sns.set_palette(sns.color_palette(colors))

    df_passive = pd.read_csv(os.path.join(data_dir, "all_passive_phase_data.csv"), sep="\t")
    plot_passive(df_passive, fig_dir)

    df_active = pd.read_csv(
        os.path.join(data_dir, "plotting_files", "indif_eta_data.csv"), sep="\t"
    )
    df_active["tmp"] = 1

    df_best = pd.read_csv(os.path.join(data_dir, "plotting_files", "best_estimates.csv"), sep="\t")

    plot_active(df_active, fig_dir)

    plot_indif_eta(df_active, pal, fig_dir)

    df_log_reg_all = pd.read_csv(
        os.path.join(data_dir, "plotting_files", "df_log_reg_all.csv"), sep="\t"
    )
    df_log_reg_subject = pd.read_csv(
        os.path.join(data_dir, "plotting_files", "df_log_reg_subjects.csv"), sep="\t"
    )

    plot_log_reg(df_active, df_log_reg_all, df_log_reg_subject, df_best, fig_dir)

    df_bayesian_all = pd.read_csv(
        os.path.join(data_dir, "plotting_files", "df_bayesian_all.csv"), sep="\t"
    )
    df_bayesian_subject = pd.read_csv(
        os.path.join(data_dir, "plotting_files", "df_bayesian_subjects.csv"), sep="\t"
    )
    plot_bayesian(df_active, df_bayesian_all, df_bayesian_subject, df_best, fig_dir)

    df_tmp = df_best[df_best.participant == "all"]
    d = {
        "0_0": list(np.random.normal(df_tmp.log_reg_best_0, df_tmp.log_reg_std_0, 1000)),
        "1_0": list(np.random.normal(df_tmp.log_reg_best_1, df_tmp.log_reg_std_1, 1000)),
    }
    df_tmp = pd.DataFrame(data=d)
    plot_heatmaps(df_tmp, df_bayesian_all, df_best, fig_dir)

