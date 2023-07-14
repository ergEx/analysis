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

Input data for the analysis is stored HERE and must be copied into the 'data' folder. The data is the output files from the experiment that record all the necessary information such as wealth trajectories, gambles, and choices.

## Synthetic data generation

It is also possible to generate synthetic data, please refer to the instructions provided in the _HERE_ to generate the data and copy it into the 'data' folder.

# How to run the code

The code is structured as a pipeline, and each stage of the pipeline performs a distinct analysis step. The entire pipeline is executed by running main.py. Configuration files using the YAML format is used to control the pipeline without the user needing to make changes in the code.

To run the code, please execute the following command in the terminal, changing the config file:

`python main.py config_file.yaml`

NOTE: The Bayesian model is not configurable from the YAML file. See detailed instruction on how to configure this _HERE_.

# Description of individual pipeline stages

## Reading Data

The first stage of the pipeline reads the input data files and prepares them for analysis. The data is saved as two files: 'all_data.csv' and 'all_data.mat'.

## Bayesian Model Estimation

The second stage of the pipeline uses the 'all_data.mat' file and estimate the parameters via the JAGS software (detaled information on JAGS installation can be found _HERE_). This stage will not run automatically, but is run by calling `runBayesian.sh` and configured in `runBayesian.m`
The results of the Bayesian model are saved in the file 'JAGS_parameter_estimation_{pooling}.mat'.

## Bracketing method

The third stage of the pipeline uses the 'all_data.csv' file and estimate the parameters using the bracketing method. It outputs two files 'bracketing_overview' and 'logistic' in both '.csv' and '.pkl' format.

# create JASP input

The fourth stage of the pipeline uses the 'JAGS_parameter_estimation_{pooling}.mat' and the 'bracketing_overview.csv' files and creates a new file called 'jasp_input.csv', which automatically updates the statistical results found in `JASP_results.jasp`.

## Creating Plots

The final stage of the pipeline creates plots based on the dataframes created in the previous stages. The plots are saved in the 'figs' subfolder.

# Setting up JAGS

JAGS is used to run the Bayesian analyses see https://sourceforge.net/projects/mcmc-jags/. We call JAGS from the MatJags library, which allows us to run JAGS models from within Matlab.

## Installing MatJags

To install MatJags, please follow the detailed installation instructions found in the MatJags repository (https://github.com/msteyvers/matjags).

# Contact information

In case of questions, please contact Benjamin Skjold b.skjold@lml.org.uk
