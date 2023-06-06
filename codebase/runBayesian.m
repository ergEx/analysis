% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:

% DataSource  - set which data source is used; Simualtion (0)
%                                              Pilot (1)
%                                              Full experiment (2)
% SimVersion - set which simulation to run (only used if DataSource is simulation);
%                                              grid with varying values (1-6)
%                                              varying noise (7)
%                                              varying ground truth risk aversion, and varying noise (8)
% dataPooling - set whether to do No pooling (1)
%                                 Partial pooling (individual estimates from group level distributions) (2)
%                                 Full pooling (super individual) (3)
% inferenceMode - set whether to do parameter estimation (1) or Bayesian model comparison (2)
% whichJAGS     - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals    - sets the order of qualities to run
% doParallel    - whether to run chains in parallel

% The idea is that this is written into by the user, then called by a
% cluster job via the terminal:

%%Specify startpath
restoredefaultpath
[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
addpath(fullfile(startDir,'/Bayesian_utils'));

%% Specify variables
dataSource = 1;
simVersion = 1;
dataPooling = 1:3;
inferenceMode = 1;
whichJAGS = 1;
whichQuals = 1;
doParallel = 0;

%% Call setHLM
setBayesian(dataSource,simVersion,dataPooling,inferenceMode,whichJAGS,whichQuals,doParallel,startDir)
