import sys
import time
import os
import yaml
from codebase import bracketing_method, create_JASP_input, create_plots, readingdata, utils, runBayesAnalysis


def make_shell(dataSource, inferenceMode, simVersion):
    import datetime
    now=datetime.datetime.now()
    now.isoformat()

    script_path = os.path.dirname(__file__)


    #if not os.path.isdir('shellscripts'):
    #    os.makedirs('shellscripts')
    print(dataSource)
    if dataSource == '0_simulation':
        dataSource = 0
    elif dataSource == '1_pilot':
        dataSource = 1
    elif dataSource == '2_full_data':
        dataSource = 2

    shellname = f'runBayes_{dataSource}_{inferenceMode}_{simVersion}.sh'

    shellparams = {'date': now, 'dataSource': dataSource, 'simVersion': simVersion, 
                   'inferenceMode': inferenceMode,
                'runBayesPath': os.path.abspath(os.path.join(script_path, 'codebase/'))}

    tmp = """#!/bin/bash
    #SBATCH --partition=HPC
    # Created {date}
    module load matlab
    matlab -nodesktop -nojvm -nosplash -r "addpath('{runBayesPath}'); runBayesian({dataSource}, {simVersion}, {inferenceMode}); exit;"
    """.format(**shellparams)

    return tmp, shellname


def bayesian_method(config, inferenceMode):
    import subprocess

    if config['bayesian method']['run']:
        response = input(f'\nThe Bayesian models are run from shell using slurm.'
                         +'\nTo run these you can therefore quit this pipeline and run it seperately.'
                        + '\n Or run it in a special inference mode'
                        + 'Do you want to continue without running the Bayesian models or'
                        + ' run the Bayesian models in mode? ([y]/n/s): ').lower()
        if response == 'n' or response == 'no':
            sys.exit()
        elif response =='y':
            print('Continuing without re-running the Bayesian models')
        elif response =='s':
            try:
                simversion = config['sim_version']
            except:
                print("Assuming no simversion")
                simversion = 0

            shell_script, shell_name = make_shell(config['data_variant'], inferenceMode,
                                                  simversion)

        script = os.path.join(os.path.abspath('sh_scripts'), shell_name)

        with open(script,"w+") as f:
            f.writelines(shell_script)

        subprocess.call(f'source {script}', shell=True)

def main():
    config_file = utils.get_config_filename(sys.argv)

    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not os.path.isdir('sh_scripts'):
        os.makedirs('sh_scripts')

    data_type = config["data_type"]
    data_variant = config["data_variant"]

    if not os.path.isdir(config['figure directory']):
        os.makedirs(config['figure directory'])

    start_time = time.time()
    print(f"\n--- {time.ctime(start_time)} --- ")
    print(f"\nSTARTING ANALYSIS")
    print(f"Data: {data_type} \nVariant: {data_variant}")

    readingdata.main(config_file)
    bayesian_method(config, 1)
    bracketing_method.main(config_file)
    create_JASP_input.main(config_file)
    runBayesAnalysis.main(config_file)
    create_plots.main(config_file)

    print(f"\n--- Code ran in {(time.time() - start_time):.2f} seconds ---")


if __name__ == "__main__":
    main()
