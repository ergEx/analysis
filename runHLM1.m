% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:

% dataMode    - set whether to simulate data or estimate based on choice data; Choice data (2) or no choice data (2)
% synthMode   - sets how to simulate data (only relevant for no choice data); (1) Pure additive agents
%                                                                             (2) Pure Multiplicative agents
%                                                                             (3) Condition specific agents (additive and multiplicative)
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% doParallel  - whether to run chains in parallel
% version     - which version of experimental setup to run on; (1) Synthetic data
%                                                               (2) One gamble version
%                                                               (3) Two gamble version
%                                                               (4) Two gamble version w. wealth controls
%                                                               (5) Two gamble version w. different additive c
%                                                               (6) Two gamble version w. hidden wealth

% The idea is that this is written into by the user, then called by a
% cluster job via the terminal:

%%Specify startpath
restoredefaultpath
[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
addpath(fullfile(startDir,'/HLM_utils'));

%% Specify variables
dataMode = 1;
synthMode = 1;
whichJAGS = 1;
whichQuals = 1:1;
doParallel = 0;
version = 1;

%% Call setHLM
setHLM(dataMode,synthMode,whichJAGS,whichQuals,doParallel,startDir,version)
