import sys
import time
import warnings

import yaml

from codebase import base, create_JASP_input, create_plots, create_plotting_data, readingdata


def main():
    config_file = base.get_config_filename(sys.argv)

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    simulation_variants = config["simulation_varaints"]
    data_variant = config["data_variant"]

    for i, simulation_variant in enumerate(simulation_variants):
        print(f"\nSTARTING ANALYSIS")
        print(f"Data: {data_variant}")
        if len(simulation_variants) > 1:
            print(f"Simulation variant: {simulation_variant}")
        start_time = time.time()
        print(time.ctime(start_time))

        readingdata.main(config_file, i, simulation_variant)
        create_plotting_data.main(config_file, i, simulation_variant)
        create_plots.main(config_file, i, simulation_variant)

        print(f"--- {(time.time() - start_time)} seconds ---")


if __name__ == "__main__":
    main()
