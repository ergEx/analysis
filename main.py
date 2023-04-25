import sys
import time

import yaml

from codebase import base, bracketing_method, create_plots, readingdata


def main():
    config_file = base.get_config_filename(sys.argv)

    with open(f"config_files/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    data_type = config["data type"]
    data_variant = config["data_variant"]

    print(f"\nSTARTING ANALYSIS")
    print(f"Data: {data_type} \nVariant: {data_variant}")
    start_time = time.time()
    print(time.ctime(start_time))

    try:
        readingdata.main(config_file)
    except Exception as e:
        ValueError("Reading data failed. Error: {e}")

    try:
        bracketing_method.main(config_file)
    except Exception as e:
        ValueError("Creating plotting data failed. Error: {e}")

    try:
        create_plots.main(config_file)
    except Exception as e:
        ValueError("Creating plots failed. Error: {e}")

    print(f"--- Code ran in {(time.time() - start_time)} seconds ---")


if __name__ == "__main__":
    main()
