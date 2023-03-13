import sys
import time
import warnings

import yaml
from sklearn.exceptions import ConvergenceWarning

from codebase import base, create_JASP_input, create_plots, readingdata


def main():
    config_file = base.get_config_filename(sys.argv)

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    simulation_variants = config["simulation_varaints"]

    for i, simulation_variant in enumerate(simulation_variants):
        print("STARTING ANALYSIS")
        start_time = time.time()
        print(time.ctime(start_time))

        readingdata.main(config_file, i, simulation_variant)
        create_JASP_input.main(config_file, i, simulation_variant)
        create_plots.main(config_file, i, simulation_variant)

        print(f"--- {(time.time() - start_time)} seconds ---")


if __name__ == "__main__":
    warnings.simplefilter("ignore", "Solver terminated early.*")
    main()
