function setBayesian(dataSource,simVersion,dataPooling,inferenceMode,whichJAGS,whichQuals,doParallel,startDir)

% setHLM sets up multiple HLM models to run sequentially according to inputs

% This function takes the following inputs:

% DataSource  - set which data source is used; Simualtion (0)
%                                              Pilot (1)
%                                              Full experiment (2)
% SimVersion - set which simulation to run (only used if DataSource is simulation);
%                                              full grid (1)
%                                              varying noise (2)
%                                              varying ground truth risk aversion, and varying noise (3)
% dataPooling - set whether to do No pooling (1)
%                                 Partial pooling (individual estimates from group level distributions) (2)
%                                 Full pooling (super individual) (3)
% inferenceMode - set whether to do parameter estimation (1) or Bayesian model comparison (2)
% whichJAGS     - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals    - sets the order of qualities to run
% doParallel    - whether to run chains in parallel

%% Specifies qualities to be selected from
numRuns      = length(whichQuals);     %how many separate instances of an MCMC to run
nBurnin      = [1e2,1e3,1e4,2e4,4e4];  %from 100 to 40k
nSamples     = [5e1,5e2,5e3,1e4,2e4];  %from 50 to 20k
nChains      = [4,4,4,4,4];            %Keep this to 4
nThin        = 10;                     %thinnning factor, 1 = no thinning, 2=every 2nd etc.

%% Specifies subjects, trials and directory_name
switch dataSource
    case {0}
        switch simVersion
            case {1}, subjList = 1:(26*10); nTrials = 160; folder = '0_simulation/full_grid';
            case {2}, subjList = 1:(3*10);  nTrials = 160; folder = '0_simulation/varying_variance';
            case {3}, subjList = 1:(3*10);  nTrials = 160; folder = '0_simulation/strong_weak_signal';
        end %simVersion
    case {1}, subjList = 1:11; nTrials = 160; folder = '1_pilot'; %Pilot data
    case {2}, subjList = 1:1; nTrials = 1; folder = '2_full_data';%Full experiment data
end %dataSource

%% Runs HLMs sequentially
for i=1:numRuns
    computeBayesian(dataSource,dataPooling,inferenceMode,nBurnin(whichQuals(i)),nSamples(whichQuals(i)),nThin,nChains(whichQuals(i)),subjList,whichJAGS,doParallel,startDir,nTrials,folder)
end