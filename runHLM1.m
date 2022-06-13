% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:

% runModelNum - which models to run: (1) parameter estimation
% synthMode   - sets whether to run on real data (1), synthetic data for parameter recovery (2-5; see setHLM.m for info on differences)
% nDynamics   - number of dynamics tested
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% doParallel  - whether to run chains in parallel

% The idea is that this is written into by the user, then called by a
% cluster job via the terminal:

%%Specify startpath
restoredefaultpath
[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
addpath(fullfile(startDir,'/utils/HLM'));

%% Specify variables
runModelNum = 1;
synthMode = 1;
nDynamics = 2;
whichJAGS = 1;
whichQuals = 1:1;
doParallel = 0;

%% Call setHLM
setHLM(runModelNum,whichJAGS,whichQuals,doParallel,startDir)
