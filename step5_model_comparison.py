import sys
import time
import yaml
import subprocess
from codebase import readingdata, utils

VBA_PATH = '/mnt/projects/CPM/toolboxes/VBA-toolbox'

def main():
    config_file = utils.get_config_filename(sys.argv)

    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    data_type = config["data_type"]
    data_variant = config["data_variant"]

    if sys.argv[2] == '1':
        model_selection_type = '1'
    elif sys.argv[2] == '2':
        model_selection_type = '2'
    else:
        raise ValueError("Model selection type has to be 1 for source or 2 for sbatch.")

    start_time = time.time()
    print(f"\n--- {time.ctime(start_time)} --- ")
    print(f"\Doing model comparison no:")
    print(f"Data: {data_type} \nVariant: {data_variant}")
    shellparams = {'VBA_PATH': VBA_PATH, 'data_variant' : data_variant, 
                   'model_selection_type': model_selection_type}
    matlab_call = """
    module load matlab;
    matlab -nodesktop  -r\
    "addpath('codebase/'); addpath('{VBA_PATH}');\
    modelComparison('{data_variant}', {model_selection_type}); exit;"
    """.format(**shellparams)

    subprocess.call(matlab_call, shell=True)
 
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
    except:
        write_provenance('FAILED!!')
