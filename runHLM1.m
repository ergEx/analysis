% runHLM provides inputs to setHLM for a given run on the cluster

% It provides the following inputs when calling setHLM:

% runModelNum - which models to run: 1 Optimal eta on, starting wealth (CPH Analysis),
%                                    2 Optimal eta on, current wealth, 
%                                    3 Optimal eta off, starting wealth, 
%                                    4 Optimal eta off, current welath
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% doParallel  - whether to run chains in parallel

% The idea is that this is written into by the user, then called by a
% cluster job via the terminal:

%%Add to path; all nessesary folders (but NOT subfolders)
restoredefaultpath
[startDir,~] = fileparts(mfilename('fullpath'));%specify your starting directory here (where this script runs from)
addpath(fullfile(startDir,'..','/data'));
addpath(fullfile(startDir,'/JAGS'));
addpath(fullfile(startDir,'/matjags'));
addpath(fullfile(startDir,'/samples_stats'));

%% Specify variables
runModelNum=1;
whichJAGS=1; 
whichQuals=1:1;
doParallel=0;

%% Call setHLM
setHLM(runModelNum,whichJAGS,whichQuals,doParallel)
