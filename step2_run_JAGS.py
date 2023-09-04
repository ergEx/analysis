import sys
import time
import os
import yaml
from codebase import utils


def make_shell(dataSource, inferenceMode, simVersion, quality, model_selection_type):
    import datetime
    now = datetime.datetime.now().isoformat()

    script_path = os.path.dirname(__file__)

    print(dataSource)
    if dataSource == '0_simulation':
        dataSource = 0
    elif dataSource == '1_pilot':
        dataSource = 1
    elif dataSource == '2_full_data':
        dataSource = 2

    shellname = f'runBayes_{dataSource}_{inferenceMode}_{simVersion}.sh'

    shellparams = {'date': now, 'dataSource': dataSource, 'simVersion': simVersion,
                   'inferenceMode': inferenceMode, 'quality': quality,
                   'model_selection_type': model_selection_type,
                'runBayesPath': os.path.abspath(os.path.join(script_path, 'codebase/'))}

    tmp = """#!/bin/bash
    #SBATCH --partition=HPC
    # Created {date}
    module load matlab
    matlab -nodesktop -nojvm -nosplash -r "addpath('{runBayesPath}'); runBayesian({dataSource}, {simVersion}, {inferenceMode}, {quality}, {model_selection_type}); exit;"
    """.format(**shellparams)

    return tmp, shellname


def bayesian_method(config, inferenceMode, quality, model_selection_type, executor='sbatch'):
    import subprocess

    if config['bayesian method']['run']:
        try:
            simversion = config['sim_version']
        except:
            print("Assuming no simversion")
            simversion = 0

        shell_script, shell_name = make_shell(dataSource=config['data_variant'],
                                              inferenceMode=inferenceMode,
                                                simVersion=simversion,
                                                quality=quality,
                                                model_selection_type=model_selection_type)

        script = os.path.join(os.path.abspath('sh_scripts'), shell_name)

        with open(script,"w+") as f:
            f.writelines(shell_script)

        subprocess.call(f'{executor} {script}', shell=True)

def main():
    config_file = utils.get_config_filename(sys.argv)

    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not os.path.isdir('sh_scripts'):
        os.makedirs('sh_scripts')

    data_type = config["data_type"]
    data_variant = config["data_variant"]
    quality = config['qual']
    start_time = time.time()
    print(f"\n--- {time.ctime(start_time)} --- ")
    print(f"\nSTARTING ANALYSIS")
    print(f"Data: {data_type} \nVariant: {data_variant}")

    if sys.argv[2] == '1':
        inferenceMethod = 1
    elif sys.argv[2] == '2':
        inferenceMethod = 2
    else:
        raise ValueError("Inference method, second argument has to be 1 or 2")

    if sys.argv[3] == '1':
        model_selection_type = '1'
    elif sys.argv[3] == '2':
        model_selection_type = '2'
    else:
        raise ValueError("Model selection type has to be 1 for source or 2 for sbatch.")

    if sys.argv[4] == '1':
        executor = 'source'
    elif sys.argv[4] == '2':
        executor = 'sbatch'
    else:
        raise ValueError("Executor has to be 1 for source or 2 for sbatch.")

    bayesian_method(config=config,
                    inferenceMode=inferenceMethod,
                    quality=quality,
                    model_selection_type=model_selection_type,
                    executor=executor)

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
