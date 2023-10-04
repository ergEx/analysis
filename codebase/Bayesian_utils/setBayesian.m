function setBayesian(dataSource,simVersion,dataPooling,inferenceMode,model_selection_type,whichJAGS,whichQuals,doParallel,startDir,seedChoice)

% setHLM sets up multiple HLM models to run sequentially according to inputs

% This function takes the following inputs:

% DataSource  - set which data source is used; Simualtion (0)
%                                              Pilot (1)
%                                              Full experiment (2)
% SimVersion - set which simulation to run (only used if DataSource is simulation);
%                                              grid with varying values (1-6)
%                                              varying noise (7)
%                                              varying ground truth risk aversion, and varying noise (8)
% dataPooling - set whether to do No pooling (1)
%                                 Partial pooling (individual estimates from group level distributions) (2)
%                                 Full pooling (super individual) (3)
% inferenceMode - set whether to do parameter estimation (1) or Bayesian model comparison (2)
% whichJAGS     - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals    - sets the order of qualities to run
% doParallel    - whether to run chains in parallel

%% Specifies qualities to be selected from
numRuns      = length(dataPooling);     %how many separate instances of an MCMC to run
nBurnin      = [0,0,0,0,0];
manualBurnin = [1e1,1e3,1e4,2e4,4e4];  %from 100 to 40k
nSamples     = [5e1,5e2,5e3,1e4,2e4];  %from 50 to 20k
nChains      = [2,4,4,4,4];            %Keep this to 4
nThin        = 1;                      %thinnning factor, 1 = no thinning, 2=every 2nd etc.

%% Specifies subjects, trials and directory_name
switch dataSource
    case {0}
        switch simVersion
            case {1}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/eta_n05';
            case {2}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/eta_00';
            case {3}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/eta_05';
            case {4}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/eta_10';
            case {5}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/eta_15';
            case {6}, subjList = 1:(1*10); nTrials = 160; folder = '0_simulation/grid/time_optimal';
            case {7}, subjList = 1:(2*10); nTrials = 160; folder = '0_simulation/varying_variance';
            case {8}, subjList = 1:(2*10); nTrials = 160; folder = '0_simulation/strong_weak_signal';
        end %simVersion
    case {1}, subjList = 1:11; nTrials = 160; folder = '1_pilot'; %Pilot data
    case {2}, subjList = 1:4; nTrials = 165; folder = '2_full_data';%Full experiment data
end %dataSource

%% Runs HLMs sequentially
for i=dataPooling
    computeBayesian(dataSource,i,inferenceMode,model_selection_type,nBurnin(whichQuals),nSamples(whichQuals) + manualBurnin(whichQuals) ,nThin,nChains(whichQuals),subjList,whichJAGS,doParallel,startDir,nTrials,folder,seedChoice)
end
