function runBayesian(dataSource, simVersion, inferenceMode, whichQuals, model_selection_type, dataPooling, whichJAGS)
% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:

% dataSource  - set which data source is used; Simulation (0)
%                                              Pilot (1)
%                                              Full experiment (2)
% simVersion - set which simulation to run (only used if DataSource is simulation);
%                                              grid with varying values (1-6)
%                                              varying noise (7)
%                                              varying ground truth risk aversion, and varying noise (8)
% inferenceMode - set whether to do parameter estimation (1) or Bayesian model comparison (2)
% whichQuals    - sets the order of qualities to run
% model_selection_type: - 1: EUT vs EE, 2 EUT vs EE vs weakEE
% dataPooling - set whether to do No pooling (1)
%                                 Partial pooling (individual estimates from group level distributions) (2)
%                                 Full pooling (super individual) (3)
% whichJAGS     - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags

% The idea is that this is written into by the user, then called by a
% cluster job via the terminal:

%%Specify startpath
restoredefaultpath
[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
addpath(fullfile(startDir,'/Bayesian_utils'));

doParallel = 1;
seedChoice = 1;

%% Call setHLM
setBayesian(dataSource,simVersion,dataPooling,inferenceMode,model_selection_type,whichJAGS,whichQuals,doParallel,startDir,seedChoice)
