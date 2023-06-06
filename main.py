import sys
import time

import yaml
from codebase import base, bracketing_method, create_plots, readingdata


def main():
    config_file = base.get_config_filename(sys.argv)

    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    data_type = config["data_type"]
    data_variant = config["data_variant"]

    start_time = time.time()
    print(f"\n--- {time.ctime(start_time)} --- ")
    print(f"\nSTARTING ANALYSIS")
    print(f"Data: {data_type} \nVariant: {data_variant}")

    readingdata.main(config_file)
    if config['bayesian method']['run']:
        response = input(f'\nThe Bayesian models are run from "runBayesian.sh". \nTo run these you must therefore quit this pipeline and run it seperately. \nDo you want to continue without running the Bayesian models? ([y]/n): ').lower()
        if response == 'n' or response == 'no':
            sys.exit()
        else:
            print('Continuing without re-running the Bayesian models')
    bracketing_method.main(config_file)
    create_plots.main(config_file)

    print(f"\n--- Code ran in {(time.time() - start_time):.2f} seconds ---")


if __name__ == "__main__":
    main()
