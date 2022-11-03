function setHLM(inferenceMode,synthMode,whichJAGS,whichQuals,doParallel,startDir,version)

% setHLM sets up multiple HLM models to run sequentially according to inputs

% This function takes the following inputs:

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

%% Specifies qualities to be selected from
numRuns      = length(whichQuals);     %how many separate instances of an MCMC to run
nBurnin      = [1e2,1e3,1e4,2e4,4e4];  %from 100 to 40k
nSamples     = [5e1,5e2,5e3,1e4,2e4];  %from 50 to 20k
nChains      = [4,4,4,4,4];            %Keep this to 4
nThin        = 10;                     %thinnning factor, 1 = no thinning, 2=every 2nd etc.

%% Specifies subjects, trials and directory_name
switch version
    case {1}, subjList = 1:3; nTrials = 120; dataVersion = '';  %One gamble version
    case {2}, subjList = 1:3; nTrials = 120; dataVersion = '';   %Two gamble version
    case {3}, subjList = 1:3; nTrials = 120; dataVersion = '';  %Two gamble version w. wealth controls
    case {4}, subjList = 1:11; nTrials = 120; dataVersion = 'two_gamble_new_c';   %Two gamble version w. different additive c
    case {5}, subjList = 1:3; nTrials = 120; dataVersion = '';   %Two gamble version w. hidden wealth
end %version

%% Runs HLMs sequentially
for i=1:numRuns
    computeHLM(inferenceMode,synthMode,nBurnin(whichQuals(i)),nSamples(whichQuals(i)),nThin,nChains(whichQuals(i)),subjList,whichJAGS,doParallel,startDir,dataVersion,nTrials)
end