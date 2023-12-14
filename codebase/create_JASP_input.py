import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
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

    quality_dictionary = {'chains': [2,4,4,4], 'samples': [5e1,5e2,5e3,1e4,2e4], 'manual_burnin': [1e1,1e3,1e4,2e4,4e4]}
    n_agents = config["n_agents"]
    burn_in = int(quality_dictionary['manual_burnin'][config['qual'] - 1])
    n_samples = int(quality_dictionary['samples'][config['qual'] - 1] - burn_in)
    n_chains = int(quality_dictionary['chains'][config['qual'] - 1])
    n_conditions = config["n_conditions"]

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
        eta_samples = bayesian_samples["eta_i"][:,burn_in:,:,:]
        for j, subject1 in enumerate(subjects):
            for c, condition in enumerate(CONDITION_SPECS["lambd"]):
                try:
                    eta_dist = eta_samples[:, :, j, c].flatten()
                    kde = gaussian_kde(eta_dist)
                    data[f"{c}.0_{pool}"][j] = eta_dist[np.argmax(kde.pdf(eta_dist))]
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
