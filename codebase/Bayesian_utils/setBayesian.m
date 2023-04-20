function setBayesian(inferenceMode,whichJAGS,whichQuals,doParallel,startDir,dataVersion, simVersion)

% setHLM sets up multiple HLM models to run sequentially according to inputs

% This function takes the following inputs:

% inferenceMode - set whether to do parameter estimation without pooling (1)
%                                   parameter estimation with pooling allowing individual differences (2)
%                                   parameter estimation with pooing and no individual differences (super individual) (3)
% whichJAGS     - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichJAGS     - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals    - sets the order of qualities to run
% doParallel    - whether to run chains in parallel
% dataVersion   - whether to run model on simulated data (1), pilot data (2) or full data (3)
% simVersion    - if running on simulated data; n_trials = 160, n_phenotypes = 26, n_agents = 100 (1)
%                                               n_trials = 1600, n_phenotypes = 26, n_agents = 3 (2)

%% Specifies qualities to be selected from
numRuns      = length(whichQuals);     %how many separate instances of an MCMC to run
nBurnin      = [1e2,1e3,1e4,2e4,4e4];  %from 100 to 40k
nSamples     = [5e1,5e2,5e3,1e4,2e4];  %from 50 to 20k
nChains      = [4,4,4,4,4];            %Keep this to 4
nThin        = 10;                     %thinnning factor, 1 = no thinning, 2=every 2nd etc.

%% Specifies subjects, trials and directory_name
switch dataVersion
    case {1} %simulated data
        switch simVersion
            case {1}, subjList = 1:(26*100); nTrials = 160;  folder = '0_simulation/n_160';
            case {2}, subjList = 1:(26*3); nTrials = 1000; folder = '0_simulation/n_1600';
        end %simVersion
    case {2}, subjList = 1:11; nTrials = 160; folder = '1_pilot'; %Pilot data
    case {3}, subjList = 1:1; nTrials = 1; folder = '2_full_data';%Full experiment data
end %dataVersion

%% Runs HLMs sequentially
for i=1:numRuns
    computeBayesian(inferenceMode,nBurnin(whichQuals(i)),nSamples(whichQuals(i)),nThin,nChains(whichQuals(i)),subjList,whichJAGS,doParallel,startDir,nTrials,folder)
end