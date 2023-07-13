import sys
import yaml
import subprocess

from .utils import get_config_filename

def main(config_file):

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not config["bayesfactor_analysis"]["run"]:
        return

    print(f"\nRunning BayesFactor analysis")

    data_dir = config["data directory"]

    subprocess.run(f'rscript r_analyses/bayesian_t_test.R --path {data_dir}')





if __name__ == "__main__":
    config_file = get_config_filename(sys.argv)
    main(config_file)
