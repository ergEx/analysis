import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from .base import get_config_filename
from .experiment_specs import condition_specs, sub_specs
from .utils import read_Bayesian_output


def read(save_path):
    bayesian_parameter_estimation_output_file = os.path.join(
        save_path, "Bayesian_parameter_estimation.mat"
    )
    if os.path.isfile(bayesian_parameter_estimation_output_file):
        bayesian_samples = read_Bayesian_output(bayesian_parameter_estimation_output_file)
    else:
        bayesian_samples = None
        print("Bayesian parameter estimation output not found!")
    return bayesian_samples


def extract(subjects, condition_specs, bayesian_samples):
    data = {"0.0": [None] * len(subjects), "1.0": [None] * len(subjects)}
    for j, subject1 in enumerate(subjects):
        for c, condition in enumerate(condition_specs["lambd"]):
            try:
                eta_dist = bayesian_samples["eta"][:, :, j, c].flatten()
                kde = sm.nonparametric.KDEUnivariate(eta_dist).fit()
                data[f"{c}.0"][j] = kde.support[np.argmax(kde.density)]
            except Exception as e:
                pass
    return pd.DataFrame.from_dict(data)


def main(config_file, i=0, simulation_variant=""):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    save_path = config["data_folders"][0]
    run_stage = config["create_jasp"]["run"]

    if run_stage:
        SUBJECT_SPECS = sub_specs(config["data_variant"])
        subjects = SUBJECT_SPECS["id"]
        CONDITION_SPECS = condition_specs()
        bayesian_samples = read(save_path)
        c_bayesian = extract(subjects, CONDITION_SPECS, bayesian_samples)
        c_bayesian.to_csv(os.path.join(save_path, "jasp_input.csv"), sep="\t")


if __name__ == "__main__":
    config_file = get_config_filename(sys.argv)
    main(config_file)
