import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
import yaml

from .base import get_config_filename


def plot_passive(df_passive, fig_dir):
    fig_passive, ax_passive = plt.subplots(2, 1)
    for i, participant in enumerate(set(df_passive.participant_id)):
        fig_passive_subject, ax_passive_subject = plt.subplots(2, 1)
        for c, cond in enumerate(set(df_passive.eta)):
            df_tmp = df_passive.query(
                "participant_id == @participant and eta == @cond"
            ).reset_index(drop=True)
            ax_passive[c].plot(df_tmp.trial, df_tmp.wealth, color="grey")
            ax_passive_subject[c].plot(df_tmp.trial, df_tmp.wealth, color="grey")
            ax_passive_subject[c] = passive_plot_layout(ax_passive_subject[c], c)

        fig_passive_subject.suptitle(f"Wealth trajectory in learning task for participant {i+1}")
        fig_passive_subject.tight_layout()
        fig_passive_subject.savefig(os.path.join(fig_dir, f"1_passive_{i}"))
    ax_passive[0] = passive_plot_layout(ax_passive[0], 0)
    ax_passive[1] = passive_plot_layout(ax_passive[1], 1)
    fig_passive.suptitle(f"Wealth trajectory in learning task")
    fig_passive.tight_layout()
    fig_passive.savefig(os.path.join(fig_dir, f"01_passive"))


def passive_plot_layout(ax, c, title_dict={0: "Additive", 1: "Multiplicative"}):
    ax.set(title=title_dict[c], xlabel="Trial", ylabel="Wealth")

    if c == 1:
        ax.set(yscale="log", ylabel="log Wealth (log)")

    for i in [45, 45 * 2]:
        ax.axvline(x=i, linestyle="--", color="grey")
    ax.plot([], linestyle="--", color="grey", label="reset")
    ax.legend(loc="upper left", fontsize="xx-small")
    return ax


def plot_active(df_active, fig_dir):
    fig_active, ax_active = plt.subplots(2, 1)
    for i, participant in enumerate(set(df_active.participant_id)):
        fig_active_subject, ax_active_subject = plt.subplots(2, 1)
        for c, cond in enumerate(set(df_active.eta)):
            df_tmp = df_active.query(
                "participant_id == @participant and eta == @cond"
            ).reset_index(drop=True)
            ax_active[c].plot(df_tmp.trial, df_tmp.wealth, color="grey")
            ax_active_subject[c].plot(df_tmp.trial, df_tmp.wealth, color="grey")
            ax_active_subject[c] = active_plot_layout(ax_active_subject[c], c)
        fig_active_subject.suptitle(f"Wealth trajectories in gamblig task for participant {i+1}")
        fig_active_subject.tight_layout()
        fig_active_subject.savefig(os.path.join(fig_dir, f"2_active_{i}"))
    ax_active[0] = active_plot_layout(ax_active[0], 0)
    ax_active[1] = active_plot_layout(ax_active[1], 1)
    fig_active.suptitle("Wealth trajectories in gamblig task")
    fig_active.tight_layout()
    fig_active.savefig(os.path.join(fig_dir, f"02_active"))


def active_plot_layout(ax, c, title_dict={0: "Additive", 1: "Multiplicative"}):
    ax.set(title=title_dict[c], xlabel="Trial", ylabel="Wealth")

    if c == 1:
        ax.set(yscale="log", ylabel="log Wealth (log)")
    return ax


def plot_indif_eta(df_active, pal, fig_dir):
    fig_indif_eta, ax_indif_eta = plt.subplots(2, 1)
    for c, cond in enumerate(set(df_active.eta)):
        df_tmp = df_active.query("eta == @cond").reset_index(drop=True)
        pt.RainCloud(
            x="tmp",
            hue="min_max",
            y="indif_eta",
            data=df_tmp,
            ax=ax_indif_eta[c],
            bw=0.3,
            orient="h",
            palette=pal,
            alpha=0.5,
            dodge=True,
        )
        ax_indif_eta[c].legend().set_visible(False)

        ax_indif_eta[c] = indif_plot_layout(ax_indif_eta[c], c)
    fig_indif_eta.suptitle("Descriptive indifference eta")
    fig_indif_eta.tight_layout()
    fig_indif_eta.savefig(os.path.join(fig_dir, f"03_indif_eta"))
    for i, participant in enumerate(set(df_active.participant_id)):
        fig_indif_eta_subject, ax_indif_eta_subject = plt.subplots(2, 1)
        for c, cond in enumerate(set(df_active.eta)):
            df_tmp = df_active.query(
                "participant_id == @participant and eta == @cond"
            ).reset_index(drop=True)
            pt.RainCloud(
                x="tmp",
                hue="min_max",
                y="indif_eta",
                data=df_tmp,
                ax=ax_indif_eta_subject[c],
                bw=0.3,
                orient="h",
                palette=pal,
                alpha=0.5,
                dodge=True,
            )
            ax_indif_eta_subject[c].legend().set_visible(False)
            ax_indif_eta_subject[c] = indif_plot_layout(ax_indif_eta_subject[c], c)
        fig_indif_eta_subject.suptitle(f"Desctriptive indifference eta for participant {i+1}")
        fig_indif_eta_subject.tight_layout()
        fig_indif_eta_subject.savefig(os.path.join(fig_dir, f"3_indif_eta_{i}"))


def indif_plot_layout(ax, c, title_dict={0: "Additive", 1: "Multiplicative"}):
    ax.set(title=title_dict[c], xlabel="Indifference eta", ylabel="", yticks=[], xlim=[-6, 6])

    return ax


def plot_log_reg(df_active, df_log_reg_all, df_log_reg_subject, fig_dir):

    fig_log_reg, ax_log_reg = plt.subplots(2, 1)
    for c, cond in enumerate(set(df_active.eta)):
        ax_log_reg[c].fill_between(
            df_log_reg_all.loc[:, f"x_test_{c}_0"],
            df_log_reg_all.loc[:, f"confidence_lower_{c}_0"],
            df_log_reg_all.loc[:, f"confidence_upper_{c}_0"],
            where=df_log_reg_all.loc[:, f"confidence_upper_{c}_0"]
            >= df_log_reg_all.loc[:, f"confidence_lower_{c}_0"],
            facecolor="grey",
            interpolate=True,
            alpha=0.5,
            label="95 % CI",
        )
        ax_log_reg[c].plot(
            df_log_reg_all.loc[:, f"x_test_{c}_0"],
            df_log_reg_all.loc[:, f"est_{c}_0"],
            color="black",
        )
        df_tmp = df_active.query("eta == @cond")

        sns.regplot(
            x=np.array(df_tmp[df_tmp.min_max == 1].indif_eta),
            y=np.array(df_tmp[df_tmp.min_max == 1].min_max),
            fit_reg=False,
            y_jitter=0.05,
            ax=ax_log_reg[c],
            label="Upper Bound",
            scatter_kws={"alpha": 0.5, "s": 3},
        )

        sns.regplot(
            x=np.array(df_tmp[df_tmp.min_max == 0].indif_eta),
            y=np.array(df_tmp[df_tmp.min_max == 0].min_max),
            fit_reg=False,
            y_jitter=0.05,
            ax=ax_log_reg[c],
            label="Upper Bound",
            scatter_kws={"alpha": 0.5, "s": 3},
        )

        ax_log_reg[c] = log_reg_layout(ax_log_reg[c], c)
    fig_log_reg.suptitle("Logistic regression")
    fig_log_reg.tight_layout()
    fig_log_reg.savefig(os.path.join(fig_dir, f"04_log_reg"))

    for i, participant in enumerate(set(df_active.participant_id)):
        fig_log_reg_subject, ax_log_reg_subject = plt.subplots(2, 1)
        for c, cond in enumerate(set(df_active.eta)):
            ax_log_reg_subject[c].fill_between(
                df_log_reg_subject.loc[:, f"x_test_{c}_{i+1}"],
                df_log_reg_subject.loc[:, f"confidence_lower_{c}_{i+1}"],
                df_log_reg_subject.loc[:, f"confidence_upper_{c}_{i+1}"],
                where=df_log_reg_subject.loc[:, f"confidence_upper_{c}_{i+1}"]
                >= df_log_reg_subject.loc[:, f"confidence_lower_{c}_{i+1}"],
                facecolor="grey",
                interpolate=True,
                alpha=0.5,
                label="95 % CI",
            )
            ax_log_reg_subject[c].plot(
                df_log_reg_subject.loc[:, f"x_test_{c}_{i+1}"],
                df_log_reg_subject.loc[:, f"est_{c}_{i+1}"],
                color="black",
            )

            df_tmp = df_active.query("participant_id == @participant and eta == @cond")

            sns.regplot(
                x=np.array(df_tmp[df_tmp.min_max == 1].indif_eta),
                y=np.array(df_tmp[df_tmp.min_max == 1].min_max),
                fit_reg=False,
                y_jitter=0.05,
                ax=ax_log_reg_subject[c],
                label="Upper Bound",
                scatter_kws={"alpha": 0.5, "s": 3},
            )

            sns.regplot(
                x=np.array(df_tmp[df_tmp.min_max == 0].indif_eta),
                y=np.array(df_tmp[df_tmp.min_max == 0].min_max),
                fit_reg=False,
                y_jitter=0.05,
                ax=ax_log_reg_subject[c],
                label="Upper Bound",
                scatter_kws={"alpha": 0.5, "s": 3},
            )

            ax_log_reg_subject[c] = log_reg_layout(ax_log_reg_subject[c], c)
        fig_log_reg_subject.suptitle(f"Logistic regression for participant {i+1}")
        fig_log_reg_subject.tight_layout()
        fig_log_reg_subject.savefig(os.path.join(fig_dir, f"4_log_reg_{i}"))


def log_reg_layout(ax, c, title_dict={0: "Additive", 1: "Multiplicative"}):
    ax.set(
        title=title_dict[c],
        xlabel="Indifference eta",
        ylabel="",
        xlim=[-6, 6],
        yticks=[0, 0.5, 1],
        xticks=np.linspace(-5, 5, 11),
    )

    ax.axhline(y=0.5, linestyle="--", color="grey")

    ax.legend(loc="upper left", fontsize="xx-small")
    return ax


def plot_bayesian(df_active, df_bayesian_all, df_bayesian_subject, fig_dir):
    legend_dict = {0: "Additive", 1: "Multiplicative"}
    fig_bayesian, ax_bayesian = plt.subplots(1, 1)
    for c, cond in enumerate(set(df_active.eta)):
        sns.kdeplot(
            df_bayesian_all.loc[:, f"{c}_0"], ax=ax_bayesian, label=legend_dict[c], fill=True,
        )
    ax_bayesian.legend(loc="upper left", fontsize="xx-small")
    ax_bayesian = bayesian_plot_layout(ax_bayesian)
    ax_bayesian.set_title("Bayesian parameter estimation")
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
        ax_bayesian_subjects.legend(loc="upper left", fontsize="xx-small")
        ax_bayesian_subjects = bayesian_plot_layout(ax_bayesian_subjects)
        ax_bayesian_subjects.set_title(f"Bayesian parameter estimation for participant {i+1}")
        fig_bayesian_subjects.savefig(os.path.join(fig_dir, f"5_bayesian_{i}"))


def bayesian_plot_layout(ax):
    ax.set(xlabel="Riskaversion parameter", ylabel="", yticks=[], xlim=[-1, 2])

    return ax


def plot_heatmaps(df, df_bayesian_all, df_best, fig_dir):
    log_reg_heatmap = sns.jointplot(
        data=df,
        x="0_0",
        y="1_0",
        fill=True,
        bw_method=0.8,
        legend=False,
        alpha=0.5,
        height=10,
        kind="kde",
        xlim=[-0.6, 1.5],
        ylim=[-0.6, 1.5],
        marginal_kws={"common_norm": False},
    )
    sns.scatterplot(
        x="log_reg_best_0",
        y="log_reg_best_1",
        data=df_best,
        ax=log_reg_heatmap.ax_joint,
        marker="x",
    )
    log_reg_heatmap.savefig(os.path.join(fig_dir, f"6_log_reg"))

    bayesian_heatmap = sns.jointplot(
        data=df_bayesian_all,
        x="0_0",
        y="1_0",
        fill=True,
        bw_method=0.8,
        legend=False,
        alpha=0.5,
        height=10,
        kind="kde",
        xlim=[-0.6, 1.5],
        ylim=[-0.6, 1.5],
        marginal_kws={"common_norm": False},
    )
    sns.scatterplot(
        x="bayesian_best_0",
        y="bayesian_best_1",
        data=df_best,
        ax=bayesian_heatmap.ax_joint,
        marker="x",
    )

    bayesian_heatmap.savefig(os.path.join(fig_dir, f"6_bayesian"))


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

    plot_log_reg(df_active, df_log_reg_all, df_log_reg_subject, fig_dir)

    df_bayesian_all = pd.read_csv(
        os.path.join(data_dir, "plotting_files", "df_bayesian_all.csv"), sep="\t"
    )
    df_bayesian_subject = pd.read_csv(
        os.path.join(data_dir, "plotting_files", "df_bayesian_subjects.csv"), sep="\t"
    )
    plot_bayesian(df_active, df_bayesian_all, df_bayesian_subject, fig_dir)

    df_tmp = df_best[df_best.participant == "all"]
    d = {
        "0_0": list(np.random.normal(df_tmp.log_reg_best_0, df_tmp.log_reg_std_0, 1000)),
        "1_0": list(np.random.normal(df_tmp.log_reg_best_1, df_tmp.log_reg_std_1, 1000)),
    }
    df_tmp = pd.DataFrame(data=d)
    plot_heatmaps(df_tmp, df_bayesian_all, df_best, fig_dir)

