function runBayesian(dataSource, simVersion, inferenceMode, whichQuals, model_selection_type, dataPooling, whichJAGS)
% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:
% DataSource (unused)  - set which data source is used; Simualtion (0)
%                                                       Pilot (1)
%                                                       Full experiment (2)
% dataPooling - set whether to do No pooling (1)
%                                 Partial pooling (individual estimates from group level distributions) (2)
%                                 Full pooling (super individual) (3)
% SimVersion - set which simulation to run (only used if DataSource is simulation);
%                                              grid with varying values (1-6)
%                                              varying noise (7)
%                                              varying ground truth risk aversion, and varying noise (8)
% inferenceMode - set whether to do parameter estimation (1)
%                                   Bayesian model comparison of three different models (2)
%                                   Bayesian model comparison of data pooling (2)
% whichQuals    - specifies the number of samples to run
% model_selection_type - set which type of model selection to perform (only used for inferencemode == 2):
%                                 - Flat prior for EUT and EE
%                                 - Flat prior for all three models
%                                 - Parameter estimation for EUT model
%                                 - Parameter estimation for EE model
%                                 - Parameter estimation for EE2 model
% dataPooling - set whether to do No pooling (1)
%                                 Partial pooling (individual estimates from group level distributions) (2)
%                                 Full pooling (super individual) (3)
% whichJAGS     - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags

%%Specify startpath
restoredefaultpath
[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
addpath(fullfile(startDir,'/Bayesian_utils'));

%% Specify variables
% dataSource = 1;
% simVersion = 1;
% dataPooling = [2, 1, 3];
% inferenceMode = 1;
% model_selection_type = 1;
%whichJAGS = 1;
%whichQuals = 1;
doParallel = 0;
seedChoice = 1;

%% Call setHLM
setBayesian(dataSource,simVersion,dataPooling,inferenceMode,model_selection_type,whichJAGS,whichQuals,doParallel,startDir,seedChoice)
