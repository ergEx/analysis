% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:

% inferenceMode - set whether to do patameter estimation (1) or model selection (2)
% synthMode     - sets how to simulate data ; (1) Real data
%                                             (2) Simulated data
% whichJAGS     - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals    - sets the order of qualities to run
% doParallel    - whether to run chains in parallel
% version       - which version of experimental setup to run on;(1) One gamble version
%                                                               (2) Two gamble version
%                                                               (3) Two gamble version w. wealth controls
%                                                               (4) Two gamble version w. different additive c
%                                                               (5) Two gamble version w. hidden wealth

% The idea is that this is written into by the user, then called by a
% cluster job via the terminal:

%%Specify startpath
restoredefaultpath
[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
addpath(fullfile(startDir,'/HLM_utils'));

%% Specify variables
inferenceMode = 1;
synthMode = 2;
whichJAGS = 1;
whichQuals = 3:3;
doParallel = 0;
version = 4;

%% Call setHLM
setHLM(inferenceMode,synthMode,aggregationMode,whichJAGS,whichQuals,doParallel,startDir,version)
