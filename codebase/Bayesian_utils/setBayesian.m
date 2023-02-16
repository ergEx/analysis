function setBayesian(inferenceMode,whichJAGS,whichQuals,doParallel,startDir,dataVersion, simVersion)

% setHLM sets up multiple HLM models to run sequentially according to inputs

% This function takes the following inputs:

% inferenceMode - set whether to do patameter estimation (1) or model selection (2)
% whichJAGS     - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals    - sets the order of qualities to run
% doParallel    - whether to run chains in parallel
% dataVersion   - whether to run model on simulated data (1), pilot data (2) or full data (3)
% simVersion    - if running on simulated data; low sensitivity,  low n  (1),
%                                               high sensitivity, low n  (2),
%                                               low sensitivity,  high n (3),
%                                               high sensitivity, high n (4)

%% Specifies qualities to be selected from
numRuns      = length(whichQuals);     %how many separate instances of an MCMC to run
nBurnin      = [1e2,1e3,1e4,2e4,4e4];  %from 100 to 40k
nSamples     = [5e1,5e2,5e3,1e4,2e4];  %from 50 to 20k
nChains      = [4,4,4,4,4];            %Keep this to 4
nThin        = 10;                     %thinnning factor, 1 = no thinning, 2=every 2nd etc.

%% Specifies subjects, trials and directory_name
switch dataVersion
    case {1} %simulated data
        subjList = 1:9;
        switch simVersion
            case {1},  nTrials = 160;  folder = '0_simulation/b_0_n_0';
            case {2},  nTrials = 160;  folder = '0_simulation/b_1_n_0';
            case {3},  nTrials = 1000; folder = '0_simulation/b_0_n_1';
            case {4},  nTrials = 1000; folder = '0_simulation/b_1_n_1';
        end %simVersion
    case {2}, subjList = 1:11; nTrials = 160; folder = '1_pilot'; %Pilot data
    case {3}, subjList = 1:1; nTrials = 1; folder = '2_full_data';%Full experiment data
end %dataVersion

%% Runs HLMs sequentially
for i=1:numRuns
    computeBayesian(inferenceMode,nBurnin(whichQuals(i)),nSamples(whichQuals(i)),nThin,nChains(whichQuals(i)),subjList,whichJAGS,doParallel,startDir,nTrials,folder)
end