import itertools
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from .base import get_config_filename
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
    with open(f"config_files/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["plot_data"]["run"]:
        return

    data_dir = config["data directoty"][i]

    print(f"\nCALCULATING PLOTTING DATA")

    if not os.path.isdir(os.path.join(data_dir, "plotting_files")):
        os.makedirs(os.path.join(data_dir, "plotting_files"))

    # READING IN DATA

    if config["plot_data"]["calculate_indif_eta"]:
        df = pd.read_csv(os.path.join(data_dir, "all_active_phase_data.csv"), sep="\t")
        df = add_indif_eta(df)
        df.to_csv(os.path.join(data_dir, "plotting_files", "indif_eta_data.csv"), sep="\t")
    else:
        try:
            df = pd.read_csv(
                os.path.join(data_dir, "plotting_files", "indif_eta_data.csv"), sep="\t"
            )
        except:
            ValueError(
                f"\nLooks like you haven't calculated the indifference etas; please do that by changing in the .yaml file you use.\n"
            )

    df = df.dropna(subset=["indif_eta"])

    try:
        bayesian_samples = read_Bayesian_output(
            os.path.join(data_dir, "Bayesian_parameter_estimation.mat")
        )
        etas = bayesian_samples["eta"]
        etas_t = etas.transpose((2, 0, 1, 3))
        mu_etas = bayesian_samples["mu_eta"]
        n_samples = np.shape(etas)[0] * np.shape(etas)[1]
        run_bayesian = True
    except:
        run_bayesian = False
        print(
            f"\nLooks like you haven't run the Bayesian model yet; you can still get the indifference eta results, but you need to run the Bayesian model if you want all the results.\n"
        )
        pass

    etas_agent = config["etas"]
    tmp = list(itertools.product(etas_agent, etas_agent)) if len(etas_agent) > 1 else [None]
    phenotypes = (
        ["random"] + [f"{i[0]}x{i[1]}" for i in tmp]
        if len(etas_agent) > 1
        else ["real_participant"]
    )

    # CREATING MULTIINDEX DATAFRAMES
    phenotype_groups = (
        np.repeat(np.arange(len(phenotypes)), len(set(df.participant_id)))
        if len(phenotypes) > 1
        else []
    )

    index_logistic = pd.MultiIndex.from_product(
        [["all"] + list(set(df.participant_id)), phenotypes, [0.0, 1.0], range(1000),],
        names=["participant", "phenotype", "dynamic", "measurement"],
    )
    df_logistic = pd.DataFrame(index=index_logistic, columns=["x_test", "pred", "lower", "upper"])

    df_logistic.to_csv(os.path.join(data_dir, "plotting_files", "logistic.csv"), sep="\t")

    if run_bayesian:
        index_bayesian = pd.MultiIndex.from_product(
            [["all"] + list(set(df.participant_id)), phenotypes, [0.0, 1.0], range(n_samples),],
            names=["participant", "phenotype", "dynamic", "measurement"],
        )
        df_bayesian = pd.DataFrame(index=index_bayesian, columns=["samples"])

    index_overview = pd.MultiIndex.from_product(
        [["all"] + list(set(df.participant_id)), phenotypes, [0.0, 1.0],],
        names=["participant", "phenotype", "dynamic"],
    )
    df_overview = pd.DataFrame(
        index=index_overview,
        columns=["log_reg_decision_boundary", "log_reg_std_dev", "bayesian_decision_boundary"],
    )
    print(len(set(df.participant_id)))
    ## CALCULATING AND ADDING DATA
    for c, con in enumerate(set(df.eta)):
        # GROUP LEVEL DATA
        for p, phenotype in enumerate(phenotypes):
            print(phenotype)
            # PHENOTYPE LEVEL DATA
            df_tmp = df.query("phenotype == @phenotype and eta == @con").reset_index(drop=True)
            (x_test, pred, lower, upper, decision_boundary, std_dev,) = logistic_regression(df_tmp)
            idx = pd.IndexSlice
            df_logistic.loc[idx[f"all", phenotype, con, :], "x_test"] = x_test
            df_logistic.loc[idx[f"all", phenotype, con, :], "pred"] = pred
            df_logistic.loc[idx[f"all", phenotype, con, :], "lower"] = lower
            df_logistic.loc[idx[f"all", phenotype, con, :], "upper"] = upper

            if run_bayesian:
                if len(phenotypes) > 1:
                    tmp = etas_t[:, :, :, c]
                    tmp_flat = tmp.reshape((tmp.shape[0], -1))

                    eta_phenotype_group = tmp_flat[phenotype_groups == p]

                    df_bayesian.loc[idx[f"all", phenotype, con, :], "samples"] = np.random.normal(
                        np.mean(eta_phenotype_group.flatten()),
                        np.std(eta_phenotype_group.flatten()),
                        size=n_samples,
                    )

                    kde = sm.nonparametric.KDEUnivariate(eta_phenotype_group.flatten()).fit()
                else:
                    df_bayesian.loc[idx[f"all", phenotype, con, :], "samples"] = mu_etas[
                        :, :, c
                    ].flatten()
                    kde = sm.nonparametric.KDEUnivariate(mu_etas[:, :, c].flatten()).fit()

            df_overview.loc[
                idx[f"all", phenotype, con], "log_reg_decision_boundary"
            ] = decision_boundary
            df_overview.loc[idx[f"all", phenotype, con], "log_reg_std_dev"] = std_dev

            if run_bayesian:
                df_overview.loc[
                    idx[f"all", phenotype, con], "bayesian_decision_boundary"
                ] = kde.support[np.argmax(kde.density)]

            for i, participant in enumerate(set(df.participant_id)):
                # INDIVIDUAL LEVEL DATA
                df_tmp = df.query(
                    "phenotype == @phenotype and participant_id == @participant and eta == @con"
                ).reset_index(drop=True)
                try:
                    (
                        x_test,
                        pred,
                        lower,
                        upper,
                        decision_boundary,
                        std_dev,
                    ) = logistic_regression(df_tmp)
                    idx = pd.IndexSlice
                    df_logistic.loc[idx[participant, phenotype, con, :], "x_test"] = x_test
                    df_logistic.loc[idx[participant, phenotype, con, :], "pred"] = pred
                    df_logistic.loc[idx[participant, phenotype, con, :], "lower"] = lower
                    df_logistic.loc[idx[participant, phenotype, con, :], "upper"] = upper
                except:
                    pass

                if run_bayesian:
                    df_bayesian.loc[idx[participant, phenotype, con, :], "samples"] = etas[
                        :, :, i + p * len(set(df.participant_id)), c
                    ].flatten()
                    print(i + p * len(set(df.participant_id)))

                    kde = sm.nonparametric.KDEUnivariate(
                        etas[:, :, i + p * len(set(df.participant_id)), c].flatten()
                    ).fit()

                    df_overview.loc[
                        idx[participant, phenotype, con], "bayesian_decision_boundary"
                    ] = kde.support[np.argmax(kde.density)]

                df_overview.loc[
                    idx[participant, phenotype, con], "log_reg_decision_boundary"
                ] = decision_boundary
                df_overview.loc[idx[participant, phenotype, con], "log_reg_std_dev"] = std_dev

    df_logistic.to_csv(os.path.join(data_dir, "plotting_files", "logistic.csv"), sep="\t")
    df_logistic.to_pickle(os.path.join(data_dir, "plotting_files", "logistic.pkl"))

    if run_bayesian:
        df_bayesian.to_csv(os.path.join(data_dir, "plotting_files", "bayesian.csv"), sep="\t")
        df_bayesian.to_pickle(os.path.join(data_dir, "plotting_files", "bayesian.pkl"))

    df_overview.to_csv(os.path.join(data_dir, "plotting_files", "overview.csv"), sep="\t")
    df_overview.to_pickle(os.path.join(data_dir, "plotting_files", "overview.pkl"))


if __name__ == "__main__":
    config_file = get_config_filename(sys.argv)

    with open(f"config_files/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    main(config_file)

