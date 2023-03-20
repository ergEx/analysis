import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .utils import (
    calculate_min_v_max,
    indiference_eta,
    logistic_regression,
    read_Bayesian_output,
    wealth_change,
)


def add_indif_eta(df):

    new_info = np.zeros([df.shape[0], 6])
    new_info_col_names = ["x1_1", "x1_2", "x2_1", "x2_2", "indif_eta", "min_max"]

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
        new_info[i, 0:4] = x_updates - trial.wealth

        root, func = indiference_eta(x_updates[0], x_updates[1], x_updates[2], x_updates[3])

        if root is not None:
            new_info[i, 4] = round(root[0], 2)
            new_info[i, 5] = calculate_min_v_max(root[0], func, trial.selected_side_map)
        else:
            new_info[i, 4] = np.nan
            new_info[i, 5] = np.nan

    col_names = list(df.columns) + new_info_col_names
    df = pd.concat([df, pd.DataFrame(new_info)], axis=1)
    df.columns = col_names
    return df


def add_log_reg(df, df_log_reg, df_best, c, i=0):
    (x_test, pred, lower, upper, decision_boundary, std_dev,) = logistic_regression(df)

    df_log_reg.loc[:, f"x_test_{c}_{i}"] = x_test
    df_log_reg.loc[:, f"confidence_lower_{c}_{i}"] = lower
    df_log_reg.loc[:, f"est_{c}_{i}"] = pred
    df_log_reg.loc[:, f"confidence_uppwer_{c}_{i}"] = upper

    df_best.loc[1, f"{c}_{i}"] = decision_boundary
    df_best.loc[2, f"{c}_{i}"] = std_dev

    return df_log_reg, df_best


def add_bayes(samples, df_bayesian_subjects, df_best, c, i=0):

    df_bayesian_subjects.loc[:, f"{c}_{i}"] = samples

    kde = sm.nonparametric.KDEUnivariate(samples).fit()

    df_best.loc[3, f"{c}_{i}"] = kde.support[np.argmax(kde.density)]

    return df_bayesian_subjects, df_best


def main(data_dir):
    add_indif_eta = False
    if add_indif_eta:
        df = pd.read_csv(os.path.join(data_dir, "all_active_phase_data.csv"), sep="\t")
        df = add_indif_eta(df)
        df.to_csv(os.path.join(data_dir, "plotting_files", "indif_eta_data.csv"), sep="\t")
    else:
        df = df.read_csv(os.path.join(data_dir, "plotting_files", "indif_eta_data.csv"))

    bayesian_samples = read_Bayesian_output(
        os.path.join(data_dir, "Bayesian_parameter_estimation.mat")
    )
    N = len(set(df.participant_id))

    cols_all = (
        [f"x_test_{c}_0" for c in range(2)]
        + [f"confidence_lower_{c}_0" for c in range(2)]
        + [f"est_{c}_0" for c in range(2)]
        + [f"confidence_upper_{c}_0" for c in range(2)]
    )

    cols_subjects = (
        [f"x_test_{c}_{i}" for i in range(N) for c in range(2)]
        + [f"confidence_lower_{c}_{i}" for i in range(1, N + 1) for c in range(2)]
        + [f"est_{c}_{i}" for i in range(1, N + 1) for c in range(2)]
        + [f"confidence_upper_{c}_{i}" for i in range(1, N + 1) for c in range(2)]
    )

    df_log_reg_all = pd.DataFrame(columns=cols_all)
    df_bayesian_all = pd.DataFrame(columns=[f"{c}_0" for c in range(2)])
    df_log_reg_subjects = pd.DataFrame(columns=cols_subjects)
    df_bayesian_subjects = pd.DataFrame(
        columns=[f"{c}_{i}" for c in range(2) for i in range(1, N + 1)]
    )
    df_best = pd.DataFrame(columns=[f"{c}_{i}" for i in range(N + 1) for c in range(2)])

    df = df.dropna(subset=["indif_eta"])

    for c, con in enumerate(set(df.eta)):
        print(c)
        # log_reg
        df_tmp = df.query("eta == @con").reset_index(drop=True)
        df_log_reg_all, df_best = add_log_reg(df_tmp, df_log_reg_all, df_best, c)
        print(df_best)

        # bayesian
        eta_dist = bayesian_samples["mu_eta"][:, :, c].flatten()
        df_bayesian_all, df_best = add_bayes(eta_dist, df_bayesian_all, df_best, c)

        for i, participant in enumerate(set(df.participant_id)):
            print("sub", i)
            # log_reg
            df_tmp = df.query("participant_id == @participant and eta == @con").reset_index(
                drop=True
            )

            df_log_reg_subjects, df_best = add_log_reg(
                df_tmp, df_log_reg_subjects, df_best, c, i + 1
            )

            # bayesian
            eta_dist = bayesian_samples["eta"][:, :, i, c].flatten()
            df_bayesian_subjects, df_best = add_bayes(
                eta_dist, df_bayesian_subjects, df_best, c, i + 1
            )

    df_log_reg_subjects.to_csv(
        os.path.join(data_dir, "plotting_files", "df_log_reg_subjects.csv"), sep="\t"
    )
    df_log_reg_all.to_csv(os.path.join(data_dir, "plotting_files", "df_log_reg_all.csv"), sep="\t")
    df_bayesian_subjects.to_csv(
        os.path.join(data_dir, "plotting_files", "df_bayesian_subjects.csv"), sep="\t"
    )
    df_bayesian_all.to_csv(
        os.path.join(data_dir, "plotting_files", "df_bayesian_all.csv"), sep="\t"
    )
    df_best.to_csv(os.path.join(data_dir, "plotting_files", "best_estimates.csv"), sep="\t")


if __name__ == "__main__":
    print(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "1_pilot")
    )
    main(data_dir)
