import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from .experiment_specs import condition_specs, sub_specs
from .utils import get_config_filename, read_Bayesian_output


def main(config_file):

    if not config["create JASP file"]["run"]:
        return

    print(f"\nCREATING JASP FILE")

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    data_dir = config["data directoty"]

    SUBJECT_SPECS = sub_specs(config["data_variant"])
    subjects = SUBJECT_SPECS["id"]
    CONDITION_SPECS = condition_specs()

    data = {"0.0_partial_pooling": [None] * len(subjects), "1.0_partial_pooling": [None] * len(subjects),
            "0.0_no_pooling": [None] * len(subjects), "1.0_no_pooling": [None] * len(subjects)}

    #bracketing method
    bracketing_overview = pd.read_csv(os.path.join(data_dir, "bracketing_overview.csv"), sep = '\t')
    data['0.0_bracketing'] = bracketing_overview.query('participant != "all" and dynamic == 0.0').reset_index(drop=True).log_reg_decision_boundary
    data['1.0_bracketing'] = bracketing_overview.query('participant != "all" and dynamic == 1.0').reset_index(drop=True).log_reg_decision_boundary

    #bayesian method
    for pool in ['no_pooling', 'partial_pooling']:
        bayesian_samples = read_Bayesian_output(
                    os.path.join(data_dir, f"Bayesian_JAGS_parameter_estimation_{pool}.mat")
                )
        for j, subject1 in enumerate(subjects):
            for c, condition in enumerate(CONDITION_SPECS["lambd"]):
                try:
                    eta_dist = bayesian_samples["eta"][:, :, j, c].flatten()
                    kde = sm.nonparametric.KDEUnivariate(eta_dist).fit()
                    data[f"{c}.0_{pool}"][j] = kde.support[np.argmax(kde.density)]
                except Exception as e:
                    pass

    df = pd.DataFrame.from_dict(data)

    df.to_csv(os.path.join(data_dir, "jasp_input.csv"), sep="\t")


if __name__ == "__main__":
    config_file = get_config_filename(sys.argv)
    main(config_file)
