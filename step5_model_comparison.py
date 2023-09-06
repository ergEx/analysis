import sys
import time
import yaml
import subprocess
from codebase import readingdata, utils

VBA_PATH = ''

def main():
    config_file = utils.get_config_filename(sys.argv)

    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    data_type = config["data_type"]
    data_variant = config["data_variant"]

    start_time = time.time()
    print(f"\n--- {time.ctime(start_time)} --- ")
    print(f"\Doing model comparison no:")
    print(f"Data: {data_type} \nVariant: {data_variant}")
    shellparams = {'VBA_PATH': VBA_PATH}
    matlab_call = """
    module load matlab;
    matlab -nodesktop -nojvm -nosplash -r\
    "addpath('codebase/'); addpath('{VBA_PATH}');\
    modelComparison(); exit;"
    """.format(**shellparams)

    print(matlab_call)
 
    print(f"\n--- Code ran in {(time.time() - start_time):.2f} seconds ---")


if __name__ == "__main__":
    import sys
    from codebase.utils import write_provenance

    command = '\t'.join(sys.argv)
    print(sys.argv)
    write_provenance(command)
    try:
        main()
    #   write_provenance('executed successfully')
    except:
        write_provenance('FAILED!!')
