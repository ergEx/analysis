_Note: currently requires python=3.9_

# Code repository for all analysis associated with the ErgEx experiment

This repository contains all code (Python v3.9, Matlab R2021b, JAGS v4.03) used to produce data analysis and figures for the ErgEx experiment. Fundamentally the code estimate riskpreferences under isoelastic utility for agents/participants playing the ErgEx game.

# Prerequisites

This code currently requires python=3.9 to run. A requirements.txt file is provided containing all other required modules.

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

The second stage of the pipeline uses the JAGS software to estimate the parameters of a Bayesian model (detaled information on JAGS installation can be found _HERE_). This stage is optional and can be omitted if the Bayesian model results are not needed. The results of the Bayesian model are saved in the file 'bayesian_parameter_estimation.mat'.

## Creating Plotting Files

The third stage of the pipeline creates dataframes that are used to create various plots. These dataframes are saved as both .csv and .pcl files in the 'plotting_files' subfolder under the 'data' folder.

## Creating Plots

The final stage of the pipeline creates plots based on the dataframes created in the previous stage. The plots are saved in the 'figs' subfolder.

Note that the Bayesian model must be run after the data has been read if one wants to include them in the analysis.

# Setting up JAGS

JAGS is used to run the Bayesian analyses see https://sourceforge.net/projects/mcmc-jags/. We call JAGS from the MatJags library, which allows us to run JAGS models from within Matlab.

## Installing MatJags

To install MatJags, please follow the detailed installation instructions found in the MatJags repository (https://github.com/msteyvers/matjags).

## Modify JAGS models

To modify the parameters in the JAGS model, use the runHLM.m file located under codebase/Bayesian_utils.

- inferenceMode - set whether to do patameter estimation (1) or model selection (2)
- whichJAGS - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
- whichQuals - sets the order of qualities to run
- doParallel - whether to run chains in parallel
- dataVersion - whether to run model on simulated data (1), pilot data (2) or full data (3)
- simVersion - if running on simulated data; n_trials = 160, n_phenotypes = 26, n_agents = 100 (1)
  n_trials = 1600, n_phenotypes = 26, n_agents = 3 (2)

# Contact information

In case of questions, please contact Benjamin Skjold b.skjold@lml.org.uk
