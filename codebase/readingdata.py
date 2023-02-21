import os

import numpy as np
import pandas as pd
import scipy.io

from .experiment_specs import condition_specs, sub_specs
from .utils import add_info_to_df


def reading_participant_passive_data(
    root_path: str, subject: str, first_run: str, bids_text: str, n_passive_runs: int
):
    """Passive phase data"""
    for run in range(1, n_passive_runs + 1):
        passive_phase_data = pd.read_csv(
            os.path.join(
                root_path,
                f"sub-{subject}",
                f"ses-{first_run}",
                f"sub-{subject}_ses-{first_run}_task-passive_acq-lambd{bids_text}_run-{run}_beh.csv",
            ),
            sep="\t",
        )

        passive_phase_data = passive_phase_data.query('event_type == "WealthUpdate"').reset_index(
            drop=True
        )
        passive_phase_data = passive_phase_data.query("part == 0").reset_index(drop=True)
    return passive_phase_data


def reading_participant_active_data(
    root_path: str,
    subject: str,
    first_run: str,
    bids_text: str,
    data_variant: str,
    lambd: float,
    run: int = 1,
) -> pd.DataFrame:
    """Active phase data"""
    if data_variant == "0_simulation":
        active_phase_data = pd.read_csv(
            os.path.join(root_path, f"sim_agent_{subject}_lambd_{int(lambd)}.csv"), sep="\t",
        )
    else:
        active_phase_data = pd.read_csv(
            os.path.join(
                root_path,
                f"sub-{subject}",
                f"ses-{first_run}",
                f"sub-{subject}_ses-{first_run}_task-active_acq-lambd{bids_text}_run-{run}_beh.csv",
            ),
            sep="\t",
        )
        active_phase_data = active_phase_data.query('event_type == "WealthUpdate"').reset_index(
            drop=True
        )
    active_phase_data["wealth"] = active_phase_data["Numbers"] = np.concatenate(
        (np.array([1000]), np.array(active_phase_data.wealth))
    )[:-1]
    active_phase_data = add_info_to_df(active_phase_data)

    return active_phase_data


def reading_data(
    data_variant: str, simulation_variant: str = " ", n_agents: int = 1, n_passive_runs: int = 3
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
    print("READING DATA")

    CONDITION_SPECS = condition_specs()
    SUBJECT_SPECS = sub_specs(data_variant)

    if simulation_variant != "":
        ROOT_PATH = os.path.join(
            os.path.dirname(__file__), "..", "data", data_variant, simulation_variant
        )
    else:
        ROOT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", data_variant)

    passive_phase_df = pd.DataFrame()
    active_phase_df = pd.DataFrame()
    datadict = dict()
    for c, condition in enumerate(CONDITION_SPECS["condition"]):
        print(f"Condition {condition}")
        for i, subject1 in enumerate(SUBJECT_SPECS["id"]):

            for j in range(n_agents):
                subject = f"{j}_{subject1}" if data_variant == "0_simulation" else subject1
                subject = subject1
                print(f"Subject {subject}")

                if data_variant != "0_simulation":
                    passive_participant_df = reading_participant_passive_data(
                        root_path=ROOT_PATH,
                        subject=subject,
                        first_run=SUBJECT_SPECS["first_run"][i][c],
                        bids_text=CONDITION_SPECS["bids_text"][c],
                        n_passive_runs=n_passive_runs,
                    )

                    passive_phase_df = pd.concat([passive_phase_df, passive_participant_df])

                active_participant_df = reading_participant_active_data(
                    root_path=ROOT_PATH,
                    subject=subject,
                    first_run=SUBJECT_SPECS["first_run"][i][c],
                    bids_text=CONDITION_SPECS["bids_text"][c],
                    data_variant=data_variant,
                    lambd=CONDITION_SPECS["lambd"][c],
                )

                ##CSV
                active_phase_df = pd.concat([active_phase_df, active_participant_df])

                ##.mat
                active_participant_df.loc[
                    np.isnan(active_participant_df["indif_eta"]), "selected_side_map"
                ] = np.nan

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
                    np.array(active_participant_df["wealth"])
                )

                # Retrieve keypresses
                datadict.setdefault(f'choice{CONDITION_SPECS["txt_append"][c]}', []).append(
                    np.array(active_participant_df["selected_side_map"])
                )

        active_phase_df.to_csv(os.path.join(ROOT_PATH, "all_active_phase_data.csv"), sep="\t")
        scipy.io.savemat(
            os.path.join(ROOT_PATH, "all_active_phase_data.mat"), datadict, oned_as="row"
        )
        np.savez(os.path.join(ROOT_PATH, "all_active_phase_data.mat.npz"), datadict=datadict)

        if not passive_phase_df.empty:
            passive_phase_df.to_csv(
                os.path.join(ROOT_PATH, "all_passive_phase_data.csv"), sep="\t"
            )


if __name__ == "__main__":
    data_variant = "1_pilot"

    # amount of noice, number of trials (unused if data_variant != 0_simulation)
    simulation_variants = (
        ["n_160", "n_1000"] if data_variant == "0_simulation" else [""]
    )  # , "b_1_n_0", "b_0_n_1", "b_1_n_1"]

    # number of repeats of simulated agents (1 if data_variant != 0_simulation)
    n_agents = 100 if data_variant == "0_simulation" else 1

    for simulation_variant in simulation_variants:
        reading_data(data_variant, simulation_variant, n_agents)
