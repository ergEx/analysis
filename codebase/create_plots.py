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
    fig_passive.savefig(os.path.join(fig_dir, f"0_1_passive"))


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
    fig_active.savefig(os.path.join(fig_dir, f"0_2_active"))


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
        move=0.225,
        width_box=0.225,
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
    fig_indif_eta.savefig(os.path.join(fig_dir, f"0_3_indif_eta"))
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


def log_reg_plot(ax, df, est, c, title_dict={0: "Additive", 1: "Multiplicative"}):
    ax.fill_between(
        df.x_test.astype(float),
        df.lower.astype(float),
        df.upper.astype(float),
        where=df.upper.astype(float) >= df.lower.astype(float),
        facecolor="grey",
        interpolate=True,
        alpha=0.5,
        label="95 % CI",
    )

    ax.plot(
        df.x_test, df.pred, color="black",
    )

    ax.axvline(x=float(est.item()), ymax=0.5, linestyle="--", color="red", label="estimate")
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


def plot_regplot(ax, df, v, label_dict={0: "Lower bound", 1: "Upper bound"}):
    sns.regplot(
        x=np.array(df[df.min_max == v].indif_eta),
        y=np.array(df[df.min_max == v].min_max),
        fit_reg=False,
        y_jitter=0.05,
        ax=ax,
        label=label_dict[v],
        scatter_kws={"alpha": 0.5, "s": 3},
    )

    return ax


def plot_log_reg(df, df_logistic, df_overview, fig_dir):

    # GROUP LEVEL
    for p, phenotype in enumerate(set(df.phenotype)):
        fig_log_reg, ax_log_reg = plt.subplots(2, 1)
        for c, con in enumerate(set(df.eta)):
            idx = pd.IndexSlice
            tmp = df_logistic.loc[idx["all", phenotype, con, :]]
            est = df_overview.loc[idx["all", phenotype, con]]

            ax_log_reg[c] = log_reg_plot(
                ax_log_reg[c], tmp, est.log_reg_decision_boundary, c, fig_dir
            )
            df_tmp = df.query("phenotype == @phenotype and eta == @con")

            ax_log_reg[c] = plot_regplot(ax_log_reg[c], df_tmp, 1)
            ax_log_reg[c] = plot_regplot(ax_log_reg[c], df_tmp, 0)

            ax_log_reg[c].legend(loc="upper left", fontsize="xx-small")
        fig_log_reg.suptitle(f"Logistic regression {phenotype}")
        fig_log_reg.tight_layout()
        fig_log_reg.savefig(os.path.join(fig_dir, f"0_4_log_reg_{phenotype}.png"))

    # PARTICIPANT LEVEL
    for p, phenotype in enumerate(set(df.phenotype)):
        for i, participant in enumerate(set(df.participant_id)):
            fig_log_reg_subject, ax_log_reg_subject = plt.subplots(2, 1)
            for c, con in enumerate(set(df.eta)):
                idx = pd.IndexSlice
                tmp = df_logistic.loc[idx[participant, phenotype, con, :]]
                est = df_overview.loc[idx[participant, phenotype, con]]
                ax_log_reg_subject[c] = log_reg_plot(
                    ax_log_reg_subject[c], tmp, est.log_reg_decision_boundary, c, fig_dir
                )
                df_tmp = df.query("participant_id == @participant and eta == @con")

                ax_log_reg_subject[c] = plot_regplot(ax_log_reg_subject[c], df_tmp, 1)
                ax_log_reg_subject[c] = plot_regplot(ax_log_reg_subject[c], df_tmp, 0)

                ax_log_reg[c].legend(loc="upper left", fontsize="xx-small")

                fig_log_reg_subject.suptitle(
                    f"Logistic regression for participant {i+1} phenotype{p}"
                )

            fig_log_reg_subject.suptitle(f"Logistic regression participant {i} {phenotype}")
            fig_log_reg_subject.tight_layout()
            fig_log_reg_subject.savefig(os.path.join(fig_dir, f"4_log_reg_{i}_{phenotype}.png"))


def plot_bayesian(df, df_bayesian, fig_dir, legend_dict={0: "Additive", 1: "Multiplicative"}):
    for p, phenotype in enumerate(set(df.phenotype)):
        fig_bayesian, ax_bayesian = plt.subplots(1, 1)
        for c, con in enumerate(set(df.eta)):
            idx = pd.IndexSlice
            tmp = df_bayesian.loc[idx["all", phenotype, con, :]]
            ax_bayesian = sns.kdeplot(
                tmp.samples.astype(float), ax=ax_bayesian, label=legend_dict[c], fill=True,
            )
            xs, ys = ax_bayesian.collections[-1].get_paths()[0].vertices.T
            ax_bayesian.fill_between(xs, ys, color="red", alpha=0.05)
            mode_idx = np.argmax(ys)
            ax_bayesian.vlines(
                xs[mode_idx], 0, ys[mode_idx], ls="--", color="red", label="Prediction"
            )
        ax_bayesian.legend(loc="upper left", fontsize="xx-small")
        ax_bayesian.set(
            title=f"Bayesian parameter estimation {phenotype}",
            xlabel="Riskaversion parameter",
            ylabel="",
            yticks=[],
            xlim=[-1, 2],
        )
        fig_bayesian.savefig(os.path.join(fig_dir, f"0_5_bayesian_{phenotype}.png"))

    for p, phenotype in enumerate(set(df.phenotype)):
        for i, participant in enumerate(set(df.participant_id)):
            fig_bayesian_subjects, ax_bayesian_subjects = plt.subplots(1, 1)
            for c, con in enumerate(set(df.eta)):
                idx = pd.IndexSlice
                tmp = df_bayesian.loc[idx[participant, phenotype, con, :]]
                ax_bayesian_subjects = sns.kdeplot(
                    tmp.samples.astype(float),
                    ax=ax_bayesian_subjects,
                    label=legend_dict[c],
                    fill=True,
                )
                xs, ys = ax_bayesian_subjects.collections[-1].get_paths()[0].vertices.T
                ax_bayesian_subjects.fill_between(xs, ys, color="red", alpha=0.05)
                mode_idx = np.argmax(ys)
                ax_bayesian_subjects.vlines(
                    xs[mode_idx], 0, ys[mode_idx], ls="--", color="red", label="Prediction"
                )
            ax_bayesian_subjects.legend(loc="upper left", fontsize="xx-small")
            ax_bayesian_subjects.set(
                title=f"Bayesian parameter estimation for participant {i} {phenotype}",
                xlabel="Riskaversion parameter",
                ylabel="",
                yticks=[],
                xlim=[-1, 2],
            )
            fig_bayesian_subjects.savefig(os.path.join(fig_dir, f"5_bayesian_{i}_{phenotype}.png"))


def plot_heatmaps(df_bayesian, fig_dir, data_variant):
    df_bayesian = df_bayesian[df_bayesian.index.get_level_values(0) != "all"]
    df_dynamics = df_bayesian.unstack(level="dynamic")
    df_dynamics.columns = ["_".join(map(str, col)).strip() for col in df_dynamics.columns.values]

    df_dynamics = df_dynamics.reset_index()

    df_dynamics["samples_0.0"] = df_dynamics["samples_0.0"].astype(float)
    df_dynamics["samples_1.0"] = df_dynamics["samples_1.0"].astype(float)

    hue = "participant" if data_variant != "0_simulation" else "phenotype"
    test = sns.jointplot(
        data=df_dynamics,
        x="samples_0.0",
        y="samples_1.0",
        hue=hue,
        kind="kde",
        alpha=0.7,
        fill=True,
        xlim=[-0.5, 1.5],
        ylim=[-0.5, 1.5],
    )
    test.savefig(os.path.join(fig_dir, f"0_6_bayesian_heatmap.png"))


def main(config_file, i, simulation_variant=""):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["plots"]["run"]:
        return

    print(f"\nPLOTTING")
    data_variant = config["data_variant"]
    data_dir = config["data directoty"][i]
    fig_dir = config["figure directoty"][i]

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    colors = config["colors"]
    pal = sns.set_palette(sns.color_palette(colors))

    if data_variant != "0_simulation":
        df_passive = pd.read_csv(os.path.join(data_dir, "all_passive_phase_data.csv"), sep="\t")
        plot_passive(df_passive, fig_dir)

    df_active = pd.read_csv(
        os.path.join(data_dir, "plotting_files", "indif_eta_data.csv"), sep="\t"
    )
    df_active["tmp"] = 1

    df_overview = pd.read_pickle(os.path.join(data_dir, "plotting_files", "overview.pkl"))

    df_logistic = pd.read_pickle(os.path.join(data_dir, "plotting_files", "logistic.pkl"))

    try:
        df_bayesian = pd.read_pickle(os.path.join(data_dir, "plotting_files", "bayesian.pkl"))
        run_bayesian = True
    except:
        run_bayesian = False
        print(
            "Looks like you haven't run the Bayesian model yet; you can still get the indifference eta results, but you need to run the Bayesian model if you want all the results."
        )

    plot_active(df_active, fig_dir)

    plot_indif_eta(df_active, pal, fig_dir)

    plot_log_reg(df_active, df_logistic, df_overview, fig_dir)

    if run_bayesian:
        plot_bayesian(df_active, df_bayesian, fig_dir)

        plot_heatmaps(df_bayesian, fig_dir, data_variant)

