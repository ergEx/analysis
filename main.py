import sys
import time

import yaml

from codebase import base, create_plots, create_plotting_data, readingdata


def main():
    config_file = base.get_config_filename(sys.argv)

    with open(f"config_files/{config_file}", "r") as f:
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

        try:
            readingdata.main(config_file, i, simulation_variant)
        except Exception as e:
            ValueError("Reading data failed. Error: {e}")

        try:
            create_plotting_data.main(config_file, i, simulation_variant)
        except Exception as e:
            ValueError("Creating plotting data failed. Error: {e}")

        try:
            create_plots.main(config_file, i, simulation_variant)
        except Exception as e:
            ValueError("Creating plots failed. Error: {e}")

        print(f"--- Code ran in {(time.time() - start_time)} seconds ---")


if __name__ == "__main__":
    main()
