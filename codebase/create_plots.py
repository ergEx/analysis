import os
import sys

import pandas as pd
import seaborn as sns
import yaml

from .base import get_config_filename
from .experiment_specs import condition_specs, sub_specs
from .plotting_utils import (
    plot_parameter_estimation_all_data_as_one,
    plot_parameter_estimation_subject_wise,
    plot_simulation_overview_heatmap,
    plot_simulation_overview_scatter,
    read_relevant_files,
)


def main(config_file, i, simulation_variant):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    data_variant = config["data_variant"]
    data_folders = config["data_folders"]
    fig_folders = config["fig_folders"]
    colors = config["colors"]
    pal = sns.set_palette(sns.color_palette(colors))
    n_agents = config["n_agents"]
    stages = config["plots"]["figures"]
    etas = config["etas"]

    CONDITION_SPECS = condition_specs()
    SUBJECT_SPECS = sub_specs(data_variant, etas)
    subjects = SUBJECT_SPECS["id"]

    print(f"\nCREATING FIGURES")
    (
        passive_phase_df,
        indifference_eta_df,
        bayesian_samples_parameter_estimation,
        bayesian_samples_model_selection,
    ) = read_relevant_files(os.path.join(data_folders[i]))

    if not os.path.isdir(fig_folders[i]):
        os.makedirs(fig_folders[i])

    if stages["subject wise"]:
        print("\nPLOT SUBJECT WISE")
        plot_parameter_estimation_subject_wise(
            fig_folders[i],
            data_variant,
            subjects,
            n_agents,
            CONDITION_SPECS,
            passive_phase_df,
            indifference_eta_df,
            bayesian_samples_parameter_estimation,
            pal,
        )

    if stages["group mean"]:
        print("\nGROUP MEAN")
        plot_parameter_estimation_all_data_as_one(
            fig_folders[i],
            data_variant,
            CONDITION_SPECS,
            indifference_eta_df,
            bayesian_samples_parameter_estimation,
            pal,
        )

    if stages["simulation overview"]:
        print("\nSIMULATION OVERVIEW")
        if n_agents > 1:
            plot_simulation_overview_heatmap(
                fig_folders[i],
                indifference_eta_df,
                subjects,
                n_agents,
                CONDITION_SPECS,
                bayesian_samples_parameter_estimation,
            )
        else:
            plot_simulation_overview_scatter(
                fig_folders[i],
                indifference_eta_df,
                subjects,
                CONDITION_SPECS,
                bayesian_samples_parameter_estimation,
            )


if __name__ == "__main__":
    config_file = get_config_filename(sys.argv)
    main(config_file)
