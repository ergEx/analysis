import itertools
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
import yaml

from .base import get_config_filename
from .experiment_specs import condition_specs, sub_specs
from .utils import wealth_change


def reading_participant_passive_data(
    data_folder: str, subject: str, first_run: str, bids_text: str, n_passive_runs: int
):
    """Passive phase data"""
    passive_phase_data = pd.DataFrame()
    for run in range(1, n_passive_runs + 1):
        df = pd.read_csv(
            os.path.join(
                data_folder,
                f"sub-{subject}",
                f"ses-{first_run}",
                f"sub-{subject}_ses-{first_run}_task-passive_acq-lambd{bids_text}_run-{run}_beh.csv",
            ),
            sep="\t",
        )
        df = df.query('event_type == "WealthUpdate" and part == 0').reset_index(drop=True)
        passive_phase_data = pd.concat([passive_phase_data, df])
    return passive_phase_data


def reading_participant_active_data(
    data_folder: str,
    phenotype: str,
    subject: str,
    first_run: str,
    bids_text: str,
    data_variant: str,
    lambd: float,
    run: int = 1,
    calc_indif_eta: bool = True,
) -> pd.DataFrame:
    """Active phase data"""
    if data_variant == "0_simulation":
        active_phase_data = pd.read_csv(
            os.path.join(
                data_folder, f"sim_agent_phenotype_{phenotype}_{subject}_lambd_{int(lambd)}.csv"
            ),
            sep="\t",
        )
    else:
        active_phase_data = pd.read_csv(
            os.path.join(
                data_folder,
                f"sub-{subject}",
                f"ses-{first_run}",
                f"sub-{subject}_ses-{first_run}_task-active_acq-lambd{bids_text}_run-{run}_beh.csv",
            ),
            sep="\t",
        )
        active_phase_data = active_phase_data.query('event_type == "WealthUpdate"').reset_index(
            drop=True
        )
        active_phase_data["phenotype"] = "real_participant"
    active_phase_data["wealth_shift"] = np.concatenate(
        (np.array([1000]), np.array(active_phase_data.wealth))
    )[:-1]

    for i, ii in enumerate(active_phase_data.index):
        trial = active_phase_data.loc[ii, :]
        x_updates = wealth_change(
            x=trial.wealth_shift,
            gamma=[
                trial.gamma_left_up,
                trial.gamma_left_down,
                trial.gamma_right_up,
                trial.gamma_right_down,
            ],
            lambd=trial.eta,
        )
        active_phase_data.loc[ii, "x1_1"] = x_updates[0] - trial.wealth_shift
        active_phase_data.loc[ii, "x1_2"] = x_updates[1] - trial.wealth_shift
        active_phase_data.loc[ii, "x2_1"] = x_updates[2] - trial.wealth_shift
        active_phase_data.loc[ii, "x2_2"] = x_updates[3] - trial.wealth_shift

        active_phase_data.loc[ii, "selected_side_map"] = (
            0 if active_phase_data.loc[ii, "selected_side"] == "right" else 1
        )

        active_phase_data.loc[ii, "selected_side_map"] = (
            active_phase_data.loc[ii, "selected_side_map"] if min(x_updates) > 0 else np.nan
        )

    return active_phase_data


def reading_data(
    data_variant: str,
    data_folder: str,
    n_agents: int = 1,
    etas: list = [],
    n_passive_runs: int = 3,
) -> None:
    """
    Reads in passive and active phase data for a given design variant, simulation status, and number of passive runs. It stores a tuple of two dataframes, one for passive phase data and one for active phase data. It also stores .mat and .npz versions of the active phase data in a subdirectory within the data directory.

    Parameters:
    simulation (bool): Whether to read in data from simulations or real experiments.
    design_variant (str): The design variant of the experiment.
    n_passive_runs (int): The number of passive runs in the experiment.
    Returns:

    one. Data is stored in the relevant subdirectories within the data directory.
    """

    CONDITION_SPECS = condition_specs()
    SUBJECT_SPECS = sub_specs(data_variant, n_agents)
    phenotypes = ["random"] + list(itertools.product(etas, etas)) if len(etas) > 1 else [None]

    passive_phase_df = pd.DataFrame()
    active_phase_df = pd.DataFrame()
    datadict = dict()
    for c, condition in enumerate(CONDITION_SPECS["condition"]):
        for p, phenotype in enumerate(phenotypes):
            if len(phenotypes) > 1:
                phe = f"{phenotype[0]}x{phenotype[1]}" if phenotype != "random" else "random"
            else:
                phe = ""
            for i, subject in enumerate(SUBJECT_SPECS["id"]):
                if data_variant != "0_simulation":
                    passive_participant_df = reading_participant_passive_data(
                        data_folder=data_folder,
                        subject=subject,
                        first_run=SUBJECT_SPECS["first_run"][i][c],
                        bids_text=CONDITION_SPECS["bids_text"][c],
                        n_passive_runs=n_passive_runs,
                    )

                    passive_phase_df = pd.concat([passive_phase_df, passive_participant_df])

                active_participant_df = reading_participant_active_data(
                    data_folder=data_folder,
                    phenotype=phe,
                    subject=subject,
                    first_run=SUBJECT_SPECS["first_run"][i][c],
                    bids_text=CONDITION_SPECS["bids_text"][c],
                    data_variant=data_variant,
                    lambd=CONDITION_SPECS["lambd"][c],
                )

                ##CSV
                active_phase_df = pd.concat([active_phase_df, active_participant_df])

                ##.mat

                # Retrieve growth rates
                datadict.setdefault(f'gr1_1{CONDITION_SPECS["txt_append"][c]}', []).append(
                    np.array(active_participant_df["gamma_left_up"])
                )
                datadict.setdefault(f'gr1_2{CONDITION_SPECS["txt_append"][c]}', []).append(
                    np.array(active_participant_df["gamma_left_down"])
                )
                datadict.setdefault(f'gr2_1{CONDITION_SPECS["txt_append"][c]}', []).append(
                    np.array(active_participant_df["gamma_right_up"])
                )
                datadict.setdefault(f'gr2_2{CONDITION_SPECS["txt_append"][c]}', []).append(
                    np.array(active_participant_df["gamma_right_down"])
                )

                # Retrieve wealth changes
                datadict.setdefault(f'x1_1{CONDITION_SPECS["txt_append"][c]}', []).append(
                    np.array(active_participant_df["x1_1"])
                )
                datadict.setdefault(f'x1_2{CONDITION_SPECS["txt_append"][c]}', []).append(
                    np.array(active_participant_df["x1_2"])
                )
                datadict.setdefault(f'x2_1{CONDITION_SPECS["txt_append"][c]}', []).append(
                    np.array(active_participant_df["x2_1"])
                )
                datadict.setdefault(f'x2_2{CONDITION_SPECS["txt_append"][c]}', []).append(
                    np.array(active_participant_df["x2_2"])
                )

                # Retrive wealth
                datadict.setdefault(f'wealth{CONDITION_SPECS["txt_append"][c]}', []).append(
                    np.array(active_participant_df["wealth_shift"])
                )

                # Retrieve keypresses
                datadict.setdefault(f'choice{CONDITION_SPECS["txt_append"][c]}', []).append(
                    np.array(active_participant_df["selected_side_map"])
                )

    if data_variant != "0_simulations":
        passive_phase_df.to_csv(os.path.join(data_folder, "all_passive_phase_data.csv"), sep="\t")
    active_phase_df.to_csv(os.path.join(data_folder, "all_active_phase_data.csv"), sep="\t")
    scipy.io.savemat(
        os.path.join(data_folder, "all_active_phase_data.mat"), datadict, oned_as="row"
    )
    np.savez(os.path.join(data_folder, "all_active_phase_data.mat.npz"), datadict=datadict)


def main(config_file, i, simulation_variant):
    with open(f"config_files/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["readingdata"]["run"]:
        return

    data_dir = config["data directoty"]
    data_variant = config["data_variant"]
    n_agents = config["n_agents"]
    etas = config["etas"]

    print(f"\nREADING DATA")
    reading_data(data_variant, data_dir[i], n_agents[i], etas)
    print("\nDATA READ SUCCESFULLY")


if __name__ == "__main__":
    config_file = get_config_filename(sys.argv)

    with open(f"config_files/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    main(config_file)

