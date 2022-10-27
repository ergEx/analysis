function setHLM(dataMode,synthMode,whichJAGS,whichQuals,doParallel,startDir,version)

% setHLM sets up multiple HLM models to run sequentially according to inputs

% This function takes the following inputs:

% dataMode    - set whether to simulate data or estimate based on choice data; Choice data (2) or no choice data (2)
% synthMode   - sets how to simulate data (only relevant for no choice data); (1) Pure additive agents
%                                                                             (2) Pure Multiplicative agents
%                                                                             (3) Condition specific agents (additive and multiplicative)
% whichJAGS   - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% whichQuals  - sets the order of qualities to run
% doParallel  - whether to run chains in parallel
% version     - which version of experimental setup to run on; (1) Synthetic data
%                                                               (2) One gamble version
%                                                               (3) Two gamble version
%                                                               (4) Two gamble version w. wealth controls
%                                                               (5) Two gamble version w. different additive c
%                                                               (6) Two gamble version w. hidden wealth

%% Specifies qualities to be selected from
numRuns      = length(whichQuals);     %how many separate instances of an MCMC to run
nBurnin      = [1e2,1e3,1e4,2e4,4e4];  %from 100 to 40k
nSamples     = [5e1,5e2,5e3,1e4,2e4];  %from 50 to 20k
nChains      = [4,4,4,4,4];            %Keep this to 4
nThin        = 10;                     %thinnning factor, 1 = no thinning, 2=every 2nd etc.

%% Specifies subjects
subjList{1} = 1:10;    %Synnthetic agents
subjList{2} = 1:3;    %One gamble version
subjList{3} = 1:3;    %Two gamble version
subjList{4} = 1:3;    %Two gamble version w. wealth controls
subjList{5} = 1:3;    %Two gamble version w. different additive c
subjList{6} = 1:3;    %Two gamble version w. hidden wealth


%% Specifies nuber of trials
nTrials{1} = 120;
nTrials{2} = 120;
nTrials{3} = 120;
nTrials{4} = 120;
nTrials{5} = 120;
nTrials{6} = 120;

% Specifies dir for experimental version
dataVersion{1} = 'simulations';
dataVersion{2} = ''; %One gamble version
dataVersion{3} = ''; %Two gamble version
dataVersion{4} = ''; %Two gamble version w. wealth controls
dataVersion{5} = 'two_gamble_new_c'; %Two gamble version w. different additive c
dataVersion{6} = ''; %Two gamble version w. hidden wealth

%% Runs HLMs sequentially
for i=1:numRuns
    computeHLM(dataMode,synthMode,nBurnin(whichQuals(i)),nSamples(whichQuals(i)),nThin,nChains(whichQuals(i)),subjList{version},whichJAGS,doParallel,startDir,dataVersion{version},nTrials{version})
end