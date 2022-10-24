function setHLM(runModelNum,dataMode,whichJAGS,whichQuals,doParallel,startDir)

% setHLM sets up multiple HLM models to run sequentially according to inputs

% This function takes the following inputs:

% runModelNum - set which model to run; parameter estimation (1) or model selection (2),
% synthMode   - sets which data to run on; see below for info on differences
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% doParallel  - whether to run chains in parallel
% startDir    - root directory for the repo
% dataVersion - which experimental setup the data comes from 

%% Specifies qualities to be selected from
numRuns      = length(whichQuals);     %how many separate instances of an MCMC to run
nBurnin      = [1e2,1e3,1e4,2e4,4e4];  %from 100 to 40k
nSamples     = [5e1,5e2,5e3,1e4,2e4];  %from 50 to 20k
nChains      = [4,4,4,4,4];            %Keep this to 4
nThin        = 10;                     %thinnning factor, 1 = no thinning, 2=every 2nd etc.

%% Specifies subjects
subjList{1}  = 1:7;     %the subjects from the experiment

%% Specifies nuber of trials
nTrials{1} = 120;

%% Specifies dir for experimental version
dataVersion{1} = 'test';

%% Runs HLMs sequentially
for i=1:numRuns
    computeHLM(runModelNum,dataMode,nBurnin(whichQuals(i)),nSamples(whichQuals(i)),nThin,nChains(whichQuals(i)),subjList,whichJAGS,doParallel,startDir,dataVersion{dataMode},nTrials{version})
end