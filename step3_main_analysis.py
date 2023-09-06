import sys
import time
import os
import yaml
from codebase import bracketing_method, create_JASP_input, create_plots, utils, runBayesAnalysis


def main():
    config_file = utils.get_config_filename(sys.argv)

    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    data_type = config["data_type"]
    data_variant = config["data_variant"]

    if not os.path.isdir(config['figure directory']):
        os.makedirs(config['figure directory'])

    start_time = time.time()
    print(f"\n--- {time.ctime(start_time)} --- ")
    print(f"\nSTARTING ANALYSIS")
    print(f"Data: {data_type} \nVariant: {data_variant}")

    bracketing_method.main(config_file)
    create_JASP_input.main(config_file)
    runBayesAnalysis.main(config_file)
    create_plots.main(config_file)

    print(f"\n--- Code ran in {(time.time() - start_time):.2f} seconds ---")


if __name__ == "__main__":
    import sys
    from codebase.utils import write_provenance

    command = '\t'.join(sys.argv)
    print(sys.argv)
    write_provenance(command)
    try:
        main()
        write_provenance('executed successfully')
    except Exception as e:
        print(e)
        write_provenance('FAILED!!')
