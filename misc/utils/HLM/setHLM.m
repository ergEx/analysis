function setHLM(runModelNum,whichJAGS,whichQuals,doParallel,startDir)
% setHLM sets up multiple HLM models to run sequentially according to inputs

% This function takes the following inputs:

% runModelNum - which models to run: (1) parameter estimation
% synthMode   - sets whether to run on real data (1), synthetic data for parameter recovery (2-5; see setHLM.m for info on differences)
% nDynamics   - number of dynamics tested
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% doParallel  - whether to run chains in parallel

%% Specifies qualities to be selected from
numRuns      = length(whichQuals);     %how many separate instances of an MCMC to run
nBurnin      = [1e2,1e3,1e4,2e4,4e4];  %from 100 to 40k
nSamples     = [5e1,5e2,5e3,1e4,2e4];  %from 50 to 20k
nChains      = [4,4,4,4,4];            %Keep this to 4
nThin        = 10;                     %thinnning factor, 1 = no thinning, 2=every 2nd etc.

%% Specifies subjects
subjList  = [1:4,6:19];  %the subjects from the experiment (excludes 5 who didnt learn the stimuli)

%% Runs HLMs sequentially
for i=1:numRuns
    computeHLM(runModelNum,synthMode,nDynamics,nBurnin(whichQuals(i)),nSamples(whichQuals(i)),nThin,nChains(whichQuals(i)),subjList,whichJAGS,doParallel,startDir)
end