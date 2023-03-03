import sys

import yaml

from codebase import base, create_plots, readingdata


def main():
    config_file = base.get_config_filename(sys.argv)

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    simulation_variants = config["simulation_varaints"]

    for i, simulation_variant in enumerate(simulation_variants):
        readingdata.main(config_file, i, simulation_variant)
        create_plots.main(config_file, i, simulation_variant)


if __name__ == "__main__":
    main()
