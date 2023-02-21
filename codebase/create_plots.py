#%% # -*- coding: utf-8 -*-
import os

import pandas as pd

from .experiment_specs import condition_specs, sub_specs
from .plotting_utils import (
    plot_bayesian_model_selection_all_as_one,
    plot_bayesian_model_selection_subject_wise,
    plot_parameter_estimation_all_data_as_one,
    plot_parameter_estimation_subject_wise,
    plot_simulation_overview,
    read_relevant_files,
)

#%%
DATA_VARIANT = "0_simulation"  #'1_pilot'
PASSIVE_RESET = 45
N_PASSIVE_RUNS = 3
ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")  # os.path.join(os.getcwd())
SIM_VARS = ["n_160", "n_1000"] if DATA_VARIANT == "0_simulation" else [""]
for SIMULATION_VARIANT in SIM_VARS:
    PATH = (
        os.path.join(DATA_VARIANT, SIMULATION_VARIANT)
        if DATA_VARIANT == "0_simulation"
        else DATA_VARIANT
    )

    N_AGENTS = 100 if DATA_VARIANT == "0_simulation" else 1

    CONDITION_SPECS = condition_specs()
    SUBJECT_SPECS = sub_specs(DATA_VARIANT)
    subjects = SUBJECT_SPECS["id"]
    INDIFFERENCE_ETA_PLOT_SPECS = {"color": {0: "orange", 1: "b"}, "sign": {0: ">", 1: "<"}}
    (
        passive_phase_df,
        indifference_eta_df,
        bayesian_samples_parameter_estimation,
        bayesian_samples_model_selection,
    ) = read_relevant_files(os.path.join(ROOT_PATH, "data", PATH,))

    save_path = os.path.join(ROOT_PATH, "figs", PATH)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    #%%
    plot_parameter_estimation_subject_wise(
        save_path,
        DATA_VARIANT,
        subjects,
        N_AGENTS,
        CONDITION_SPECS,
        passive_phase_df,
        N_PASSIVE_RUNS,
        PASSIVE_RESET,
        indifference_eta_df,
        INDIFFERENCE_ETA_PLOT_SPECS,
        bayesian_samples_parameter_estimation,
    )

    #%%
    plot_parameter_estimation_all_data_as_one(
        save_path,
        DATA_VARIANT,
        CONDITION_SPECS,
        indifference_eta_df,
        INDIFFERENCE_ETA_PLOT_SPECS,
        bayesian_samples_parameter_estimation,
    )

    #%%
    plot_bayesian_model_selection_subject_wise(
        save_path, subjects, bayesian_samples_model_selection
    )

    #%%
    plot_bayesian_model_selection_all_as_one(save_path, bayesian_samples_model_selection)

    #%%
    if DATA_VARIANT == "0_simulation":
        plot_simulation_overview(
            save_path,
            indifference_eta_df,
            subjects,
            N_AGENTS,
            CONDITION_SPECS,
            bayesian_samples_parameter_estimation,
        )

