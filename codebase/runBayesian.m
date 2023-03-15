% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:

% inferenceMode - set whether to do patameter estimation (1) or model selection (2)
% whichJAGS     - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals    - sets the order of qualities to run
% doParallel    - whether to run chains in parallel
% dataVersion   - whether to run model on simulated data (1), pilot data (2) or full data (3)
% simVersion    - if running on simulated data; n_trials = 160, n_phenotypes = 26, n_agents = 100 (1)
%                                               n_trials = 1000, n_phenotypes = 26, n_agents = 3 (2)

% The idea is that this is written into by the user, then called by a
% cluster job via the terminal:

%%Specify startpath
restoredefaultpath
[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
addpath(fullfile(startDir,'/Bayesian_utils'));

%% Specify variables
inferenceMode = 1;
whichJAGS = 1;
whichQuals = 1:1;
doParallel = 0;
dataVersion = 1;

%% Call setHLM
for simVersion = 1:1
    setBayesian(inferenceMode,whichJAGS,whichQuals,doParallel,startDir,dataVersion,simVersion)
end