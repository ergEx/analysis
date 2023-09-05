import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from .experiment_specs import condition_specs, sub_specs
from .utils import get_config_filename, read_Bayesian_output


def main(config_file):

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["JASP input"]["run"]:
        return

    print(f"\nCREATING JASP FILE")

    data_dir = config["data directory"]
    try:
        input_path = config['input_path']
    except:
        input_path = data_dir

    SUBJECT_SPECS = sub_specs(config["data_type"], config["data_variant"], input_path)
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
                    eta_dist = bayesian_samples["eta_i"][:, :, j, c].flatten()
                    kde = sm.nonparametric.KDEUnivariate(eta_dist).fit()
                    data[f"{c}.0_{pool}"][j] = kde.support[np.argmax(kde.density)]
                except Exception as e:
                    pass

    df = pd.DataFrame.from_dict(data)

    #calculate distance measures

    d_h1 = np.sqrt((df['0.0_partial_pooling'] - 0) ** 2 + (df['1.0_partial_pooling'] - 1) ** 2)
    d_h0 = np.abs(df['0.0_partial_pooling'] - df['1.0_partial_pooling']) / np.sqrt(2)
    df.insert(2, 'd_h1_partial_pooling', d_h1)
    df.insert(3, 'd_h0_partial_pooling', d_h0)

    d_h1 = np.sqrt((df['0.0_no_pooling'] - 0) ** 2 + (df['1.0_no_pooling'] - 1) ** 2)
    d_h0 = np.abs(df['0.0_no_pooling'] - df['1.0_no_pooling']) / np.sqrt(2)
    df.insert(6, 'd_h1_no_pooling', d_h1)
    df.insert(7, 'd_h0_no_pooling', d_h0)

    d_h1 = np.sqrt((df['0.0_bracketing'] - 0) ** 2 + (df['1.0_bracketing'] - 1) ** 2)
    d_h0 = np.abs(df['0.0_bracketing'] - df['1.0_bracketing']) / np.sqrt(2)
    df.insert(10, 'd_h1_bracketing', d_h1)
    df.insert(11, 'd_h0_bracketing', d_h0)

    df.to_csv(os.path.join(data_dir, "jasp_input.csv"), sep="\t")


if __name__ == "__main__":
    config_file = get_config_filename(sys.argv)
    main(config_file)
