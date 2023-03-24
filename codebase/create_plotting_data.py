import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from .utils import (
    calculate_min_v_max,
    indiference_eta,
    logistic_regression,
    read_Bayesian_output,
    wealth_change,
)


def add_indif_eta(df):

    new_info = np.zeros([df.shape[0], 2])
    new_info_col_names = ["indif_eta", "min_max"]

    for i, ii in enumerate(df.index):
        trial = df.loc[ii, :]
        x_updates = wealth_change(
            x=trial.wealth,
            gamma=[
                trial.gamma_left_up,
                trial.gamma_left_down,
                trial.gamma_right_up,
                trial.gamma_right_down,
            ],
            lambd=trial.eta,
        )

        root, func = indiference_eta(x_updates[0], x_updates[1], x_updates[2], x_updates[3])

        if root is not None:
            new_info[i, 0] = round(root[0], 2)
            new_info[i, 1] = calculate_min_v_max(root[0], func, trial.selected_side_map)
        else:
            new_info[i, 0] = np.nan
            new_info[i, 1] = np.nan

    col_names = list(df.columns) + new_info_col_names
    df = pd.concat([df, pd.DataFrame(new_info)], axis=1)
    df.columns = col_names
    return df


def main(config_file, i, simulation_variant):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["plot_data"]["run"]:
        return

    data_dir = config["data directoty"][i]

    # READING IN DATA

    if config["plot_data"]["calculate_indif_eta"]:
        df = pd.read_csv(os.path.join(data_dir, "all_active_phase_data.csv"), sep="\t")
        df = add_indif_eta(df)
        df.to_csv(os.path.join(data_dir, "plotting_files", "indif_eta_data.csv"), sep="\t")
    else:
        df = pd.read_csv(os.path.join(data_dir, "plotting_files", "indif_eta_data.csv"), sep="\t")

    df = df.dropna(subset=["indif_eta"])

    bayesian_samples = read_Bayesian_output(
        os.path.join(data_dir, "Bayesian_parameter_estimation.mat")
    )

    etas = bayesian_samples["eta"]
    mu_etas = bayesian_samples["mu_eta"]

    # CREATING MULTIINDEX DATAFRAMES

    index_logistic = pd.MultiIndex.from_product(
        [["all"] + list(set(df.participant_id)), [0.0, 1.0], range(1000)],
        names=["participant", "dynamic", "measurement"],
    )
    df_logistic = pd.DataFrame(index=index_logistic, columns=["x_test", "pred", "lower", "upper"])

    index_bayesian = pd.MultiIndex.from_product(
        [
            ["all"] + list(set(df.participant_id)),
            [0.0, 1.0],
            range(np.shape(etas)[0] * np.shape(etas)[1]),
        ],
        names=["participant", "dynamic", "measurement"],
    )
    df_bayesian = pd.DataFrame(index=index_bayesian, columns=["samples"])

    index_overview = pd.MultiIndex.from_product(
        [["all"] + list(set(df.participant_id)), [0.0, 1.0], range(1)],
        names=["participant", "dynamic", "measurement"],
    )
    df_overview = pd.DataFrame(
        index=index_overview,
        columns=["log_reg_decision_boundary", "log_reg_std_dev", "bayesian_decision_boundary"],
    )

    ## CALCULATING AND ADDING DATA
    for c, con in enumerate(set(df.eta)):
        # GROUP LEVEL DATA
        df_tmp = df.query("eta == @con").reset_index(drop=True)
        (x_test, pred, lower, upper, decision_boundary, std_dev,) = logistic_regression(df_tmp)
        idx = pd.IndexSlice
        df_logistic.loc[idx["all", con, :], "x_test"] = x_test
        df_logistic.loc[idx["all", con, :], "pred"] = pred
        df_logistic.loc[idx["all", con, :], "lower"] = lower
        df_logistic.loc[idx["all", con, :], "upper"] = upper

        df_bayesian.loc[idx["all", con, :], "samples"] = etas[:, :, i, c].flatten()

        kde = sm.nonparametric.KDEUnivariate(mu_etas[:, :, c].flatten()).fit()

        df_overview.loc[idx["all", con, :], "log_reg_decision_boundary"] = decision_boundary
        df_overview.loc[idx["all", con, :], "log_reg_std_dev"] = std_dev
        df_overview.loc[idx["all", con, :], "bayesian_decision_boundary"] = kde.support[
            np.argmax(kde.density)
        ]

        for i, participant in enumerate(set(df.participant_id)):
            # INDIVIDUAL LEVEL DATA
            df_tmp = df.query("participant_id == @participant and eta == @con").reset_index(
                drop=True
            )
            (x_test, pred, lower, upper, decision_boundary, std_dev,) = logistic_regression(df_tmp)
            idx = pd.IndexSlice
            df_logistic.loc[idx[participant, con, :], "x_test"] = x_test
            df_logistic.loc[idx[participant, con, :], "pred"] = pred
            df_logistic.loc[idx[participant, con, :], "lower"] = lower
            df_logistic.loc[idx[participant, con, :], "upper"] = upper

            df_bayesian.loc[idx[participant, con, :], "samples"] = etas[:, :, i, c].flatten()

            kde = sm.nonparametric.KDEUnivariate(etas[:, :, i, c].flatten()).fit()

            df_overview.loc[
                idx[participant, con, :], "log_reg_decision_boundary"
            ] = decision_boundary
            df_overview.loc[idx[participant, con, :], "log_reg_std_dev"] = std_dev
            df_overview.loc[idx[participant, con, :], "bayesian_decision_boundary"] = kde.support[
                np.argmax(kde.density)
            ]
    df_logistic.to_csv(os.path.join(data_dir, "plotting_files", "logistic.csv"), sep="\t")
    df_logistic.to_pickle(os.path.join(data_dir, "plotting_files", "logistic.pkl"))

    df_bayesian.to_csv(os.path.join(data_dir, "plotting_files", "bayesian.csv"), sep="\t")
    df_bayesian.to_pickle(os.path.join(data_dir, "plotting_files", "bayesian.pkl"))

    df_overview.to_csv(os.path.join(data_dir, "plotting_files", "overview.csv"), sep="\t")
    df_overview.to_pickle(os.path.join(data_dir, "plotting_files", "overview.pkl"))


if __name__ == "__main__":
    print(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "1_pilot")
    )
    main(data_dir)
