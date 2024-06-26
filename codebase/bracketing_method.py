import os
import sys

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from .utils import calculate_min_v_max, get_config_filename, indiference_eta, logistic_regression, wealth_change


def add_indif_eta(df):

    new_info = np.zeros([df.shape[0], 2])
    new_info_col_names = ["indif_eta", "min_max"]

    for i, ii in tqdm(enumerate(df.index), desc='Adding indifference etas', total=len(df.index)):
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


def main(config_file):
    with open(f"{config_file}", "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    data_dir = config["data directory"]

    if not config["bracketing method"]["run"]:
        return

    if config["bracketing method"]["calculate_indif_eta"]:
        print('\nADDING INDIFFERENCE ETAS')
        df = pd.read_csv(os.path.join(data_dir, "all_active_phase_data.csv"), sep="\t")
        df = add_indif_eta(df)
        df.to_csv(os.path.join(data_dir, "all_active_phase_data_w_indif_etas.csv"), sep="\t")
        df.to_pickle(os.path.join(data_dir, "all_active_phase_data_w_indif_etas.pkl"))
    else:
        try:
            df = pd.read_csv(
                os.path.join(data_dir, "all_active_phase_data_w_indif_etas.csv"), sep="\t"
            )
        except:
            ValueError(
                "\nLooks like you haven't calculated the indifference etas; please do that by changing in the .yaml file you use.\n"
            )

    if not config["bracketing method"]["log_reg"]:
        return

    df = df.dropna(subset=["indif_eta"])

    participants = np.unique(df.participant_id)
    participants_sort = np.argsort([f'{i}'.upper() for i in participants])
    participant_list = list(participants[participants_sort])

    index_logistic = pd.MultiIndex.from_product(
        [["all"] + participant_list, [0.0, 1.0], range(1000),],
        names=["participant", "dynamic", "measurement"],
    )
    df_logistic = pd.DataFrame(index=index_logistic, columns=["x_test", "pred", "lower", "upper"])

    index_bracketing_overview = pd.MultiIndex.from_product(
        [["all"] + participant_list, [0.0, 1.0],],
        names=["participant", "dynamic"],
    )
    df_bracketing_overview = pd.DataFrame(
        index=index_bracketing_overview,
        columns=["log_reg_decision_boundary", "log_reg_std_dev"],
    )

    ## CALCULATING AND ADDING DATA
    print('\nCALCULATING LOGISTIC REGRESSION')
    for con in set(df.eta):
        # GROUP LEVEL DATA
        df_tmp = df.query("eta == @con").reset_index(drop=True)
        (x_test, pred_mean, ci_lower, ci_upper, mu, std,) = logistic_regression(df_tmp)
        idx = pd.IndexSlice
        df_logistic.loc[idx["all", con, :], "x_test"] = x_test
        df_logistic.loc[idx["all", con, :], "pred"] = pred_mean
        df_logistic.loc[idx["all", con, :], "lower"] = ci_lower
        df_logistic.loc[idx["all", con, :], "upper"] = ci_upper

        df_bracketing_overview.loc[
            idx["all", con], "log_reg_decision_boundary"
        ] = mu
        df_bracketing_overview.loc[idx["all", con], "log_reg_std_dev"] = std

        for i, participant in tqdm(enumerate(participant_list),
                                   desc='Calculating logistic regression',
                                   total=len(participant_list)):
            # INDIVIDUAL LEVEL DATA
            df_tmp = df.query(
                "participant_id == @participant and eta == @con"
            ).reset_index(drop=True)

            (x_test, pred_mean, ci_lower, ci_upper, mu, std) = logistic_regression(df_tmp)

            idx = pd.IndexSlice
            df_logistic.loc[idx[participant, con, :], "x_test"] = x_test
            df_logistic.loc[idx[participant, con, :], "pred"] = pred_mean
            df_logistic.loc[idx[participant, con, :], "lower"] = ci_lower
            df_logistic.loc[idx[participant, con, :], "upper"] = ci_upper

            df_bracketing_overview.loc[
                idx[participant, con], "log_reg_decision_boundary"
            ] = mu
            df_bracketing_overview.loc[idx[participant, con], "log_reg_std_dev"] = std

    df_logistic.to_csv(os.path.join(data_dir, "logistic.csv"), sep="\t")
    df_logistic.to_pickle(os.path.join(data_dir, "logistic.pkl"))

    df_bracketing_overview.to_csv(os.path.join(data_dir, "bracketing_overview.csv"), sep="\t")
    df_bracketing_overview.to_pickle(os.path.join(data_dir, "bracketing_overview.pkl"))

    print('\nRESULTS FROM BRACKETING METHOD SAVED SUCCESFULLY')


if __name__ == "__main__":
    config_file = get_config_filename(sys.argv)

    main(config_file)
