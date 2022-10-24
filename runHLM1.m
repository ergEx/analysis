% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:

% runModelNum - set which model to run; parameter estimation (1) or model selection (2),
% synthMode   - sets which data to run on; see setHLM.m for info on differences,
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% doParallel  - whether to run chains in parallel

% The idea is that this is written into by the user, then called by a
% cluster job via the terminal:

%%Specify startpath
restoredefaultpath
[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
addpath(fullfile(startDir,'/HLM_utils'));

%% Specify variables
runModelNum = 1;
dataMode = 1;
whichJAGS = 1;
whichQuals = 1:1;
doParallel = 0;

%% Call setHLM
setHLM(runModelNum,dataMode,whichJAGS,whichQuals,doParallel,startDir)
