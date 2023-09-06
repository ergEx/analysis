_Note: currently requires python=3.9_

# Code repository for all analysis associated with the ErgEx experiment

This repository contains all code (Python v3.9, Matlab R2021b, JAGS v4.03) used to produce data analysis and figures for the ErgEx experiment. Fundamentally the code estimate riskpreferences under isoelastic utility for agents/participants playing the ErgEx game see details on experiment here: https://github.com/ergEx/experiment.

# Prerequisites

This code currently requires python=3.9 to run. A requirements.txt file is provided containing all other required modules. To do the analysis in R, we advise to use conda and to create a new environment using the environment.yml file.
This will create an environment including both R and Python in the required versions. Note: running the Bayes Factor Design Analysis will require installing the package from GitHub, so it is done inside the script.

Install and check if environment already exists:
`conda env create -f environment.yml || conda env update -f environment.yml`.

# Running BFDA:

To run the Bayes Factor Design Analysis after installing the environment use in the main folder:
`rscript r_analyses/ergEx_rr_nhb_bfda.R`. When run the first time the BFDA package will be installed. 
The figures referred to in the paper will be created inside the `r_analyses` folder.

# Data

Input data for the analysis is stored HERE and should be copied into the 'data' folder. The data is the output files from the experiment that record all the necessary information such as wealth trajectories, gambles, and choices.

## Synthetic data generation

It is also possible to generate synthetic data, please refer to the instructions provided in the _HERE_ to generate the data and copy it into the 'data' folder.

# Workflow

## Step 0: Create supporting plots for paper

This step does not require any data. 

`python step0_create_supporting_plots.py`

## Step 1: Read data

`python step1_read_data config_files/config_1_pilot.yaml`

In this step the data is collected from the location specified in
`configs[input_path]`. If it is not specifed `configs[data_directory]` is used.
The data will be saved in `configs[data_directory]`, which is relative to the 
`data\` directory of the repository.

The first stage of the pipeline reads the input data files and prepares them for analysis. The data is saved as two files: 'all_data.csv' and 'all_data.mat'.


## Step 2: Run jags


The second stage of the pipeline uses the 'all_data.mat' file and estimate the parameters via the JAGS software (detailed information on JAGS installation can be found _HERE_). 
The results of the Bayesian model are saved in the file 'JAGS_parameter_estimation_{pooling}.mat'.

`python step2_run_JAGS.py config_files/config_1_pilot.yaml 1 1 1`

The arguments in order:
1. the config file to be used
2. Sets the `inferenceMode`, it can be `1` or `2` and decides if to perform model inversion for parameter estimation or model selection. In mode `3` it is doing model selection between pooling methods.
3. Sets the `model_selection_type`, it can be `1` for EE vs EUT or `2`, which further includes weak EUT.
4. This sets the submission method. It can be `1` for simply sourcing the shell script or `2` for commiting the script via SLURM.
5. Which JAGS, set this to run multiple JAGS models at the same time.


For a full analysis of the data set, you need thus to run the following three for each dataset:

`python step2_run_JAGS.py config_files/config_1_pilot.yaml 1 1 1 1`

`python step2_run_JAGS.py config_files/config_1_pilot.yaml 2 1 1 1`

`python step2_run_JAGS.py config_files/config_1_pilot.yaml 2 2 1 1`

`python step2_run_JAGS.py config_files/config_1_pilot.yaml 3 2 1 1`

This script creates a shell script in `sh_scripts` which will then be run by the program. The following `configs[data_type]`, `configs[data variant]` and `configs[qual]` are important. 

Note further, that not all arguments from the config files are used! For a new data set make sure that you have the correct name pairing and parameters in `set_Bayesian.m` lines 30 ff.

```{matlab}
switch dataSource
    case {0}
        switch simVersion
            case {1}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/eta_n05';
            case {2}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/eta_00';
            case {3}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/eta_05';
            case {4}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/eta_10';
            case {5}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/eta_15';
            case {6}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/time_optimal';
            case {7}, subjList = 1:(2*10); nTrials = 160; folder = '0_simulation/varying_variance';
            case {8}, subjList = 1:(2*10); nTrials = 160; folder = '0_simulation/strong_weak_signal';
        end %simVersion
    case {1}, subjList = 1:11; nTrials = 160; folder = '1_pilot'; %Pilot data
    case {2}, subjList = 1:1; nTrials = 1; folder = '2_full_data';%Full experiment data
end %dataSource
```

## Step3: Create figures and run further analysis:

Again this requires the configs file and creates most of the other figures that are displayed in the paper (and more) and performs further analyses.

`python step3_main_analysis.py config_files/1_pilot.yaml`

There are a few substeps:
**Bracketing method**

This stage of the pipeline uses the 'all_data.csv' file and estimate the parameters using the bracketing method. It outputs two files 'bracketing_overview' and 'logistic' in both '.csv' and '.pkl' format.

**Create JASP input**

We are not using JASP anymore, but it is here for posterity and we call an
Rscript on the outputs.
This step uses the 'JAGS_parameter_estimation_{pooling}.mat' and the 'bracketing_overview.csv' files and creates a new file called 'jasp_input.csv'.

**Creating Plots**

The final step of the pipeline creates plots based on the dataframes created in the previous stages. The plots are saved in the 'figs' subfolder.

## Step4: Create grid figure

If you have run the simulations using JAGS for configs 1 - 6, you can now create the simulation grid figure, it uses data stored under `data/0_simulation/grid`:

`python step4_plot_grid.py`

## Step5: Model comparison

This step runs the model comparison. You will have to set the path to the (VBA_toolbox)[https://github.com/MBB-team/VBA-toolbox] in `step5_model_comparison` by changing the `VBA_PATH` variable.

This step has two modes, one for each model comparison approach:

`python step5_model_comparison.py config_files/config_1_pilot.yaml 1`

`python step5_model_comparison.py config_files/config_1_pilot.yaml 2`


## Step6: Renaming figures

This step is mostly for convenience, it moves and renames the figures that are shown in the manuscript.

`python step6_plots_to_paper.py`

# Setting up JAGS

JAGS is used to run the Bayesian analyses see https://sourceforge.net/projects/mcmc-jags/. We call JAGS from the MatJags library, which allows us to run JAGS models from within Matlab.

## Installing MatJags

To install MatJags, please follow the detailed installation instructions found in the MatJags repository (https://github.com/msteyvers/matjags).

# Contact information

In case of questions, please contact Benjamin Skjold b.skjold@lml.org.uk
