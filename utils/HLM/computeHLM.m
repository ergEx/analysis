function computeHLM(runModelNum,synthMode,nDynamics,nBurnin,nSamples,nThin,nChains,subjList,whichJAGS,doParallel,startDir)
%% Hiercharchical Latent Mixture (HLM) model
% This is a general script for running several types of hierarchical
% bayesian model via JAGS. It can run hiearchical latent mixture models in
% which different utility models can be compared via inference on the model
% indicator variables, and it can run without latent mixtures of models for
% instance in order to estimate parameters of a given utility model. It
% takes as input the following:

% runModelNum - a number which selects which JAGS model to run.
% synthMode   - sets whether to run on real data (1), synthetic data for parameter recovery (2-5; see setHLM.m for info on differences)
% nDynamics   - number of dynamics tested
% nBurnin     - a number which specifies how many burn in samples to run.
% nSamples    - a number specifying the number of samples to run
% nThin       - a number specifying the thinnning number
% nChains     - number of chains
% subjList    - list of subject numbers to include
% whichJAGS   - sets which copy of matjags to run
% runPlots    - sets whether to plot data/illustrations(1) or to suppress (0)
% synthMode   - sets whether to simulate data (0), run on real data (1), synthetic data for parameter recovery (2-10)
% doParallel  - sets whether to run chains in parallel
% startDir    - path from which everything is relative to

%% Set and add paths
jagsDir = fullfile(startDir,'/utils/HLM/JAGS');
dataDir = fullfile(startDir,'/data');
samplesDir = fullfile(startDir,'/HLM_samples_stats');
addpath(jagsDir);
addpath(dataDir);
addpath(samplesDir);

%% Choose & load data
switch synthMode
    case {1}, dataSource = 'all_data_experiment.mat';    %Real data
    case {2}, dataSource = 'all_data_synthetic_0d0.mat'; %Synthetic data type specified in title
end
load(fullfile(dataDir,dataSource))

%% Choose JAGS file
switch runModelNum
    case {1}, modelName = 'JAGS_parameter_estimation.txt';
end

%% Set key variables
nTrials = 299;
doDIC=0;                     %compute Deviance information criteria? This is the hierarchical equivalent of an AIC, the lower the better
nSubjects=length(subjList);  %number of agents

%% Set bounds of hyperpriors
%hard code the upper and lower bounds of hyperpriors, typically uniformly
%distributed in log space. These values will be imported to JAGS.

%beta - prior on log since cannot be less than 0; note same bounds used for independent priors on all utility models
muLogBetaL=-2.3;muLogBetaU=3.4;muLogBetaM=(muLogBetaL+muLogBetaU)/2; %bounds on mean of distribution log beta
sigmaLogBetaL=0.01;sigmaLogBetaU=sqrt(((muLogBetaU-muLogBetaL)^2)/12);sigmaLogBetaM=(sigmaLogBetaL+sigmaLogBetaU)/2;%bounds on the std of distribution of log beta

%eta
muEtaL=-2.5;muEtaU=2.5;muEtaM=(muEtaL+muEtaU)/2;%bounds on mean of distribution of eta
sigmaEtaL=0.01;sigmaEtaU=sqrt(((muEtaU-muEtaL)^2)/12);sigmaEtaM=(sigmaEtaL+sigmaEtaU)/2;%bounds on std of eta

%% Print information for user
disp('**************');
disp(['JAGS script: ', modelName])
disp(['on: ', dataSource])
disp(['started: ',datestr(clock)])
disp(['MCMC number: ',num2str(whichJAGS)])
disp('**************');

%% Initialise matrices
%initialise matrices with nan values of size subjects x dynamics x trials
dim = nan(nSubjects,nDynamics,nTrials);          %specify the dimension
choice = dim;                                    %initialise choice data matrix
w = dim                                          %initialise wealth matrix
w1_1 = dim; w1_2 = dim; w2_1 = dim; w2_2 = dim;  %initialise wealth_change matrices

%% Compile choice & gamble data
% Jags cannot deal with partial observations, so we need to specify gamble info for all nodes. This doesn't change anything.

%We allow all agents to have different session length. Therefore we add random gambles to pad out to maxTrials. This
%allows jags to work since doesn't work for partial observation. This does not affect
%parameter estimation. nans in the choice data are allowed as long as all covariates are not nan.
for i = 1:nSubjects
    for c = 1:nDynamics
        trialInds=1:nTrials;
        switch c
            case {1} %eta = 0
                choice(i,c,trialInds)=choice_add{i}(trialInds);
                w1_1(i,c,trialInds)=x1_add{i}(trialInds);
                w1_2(i,c,trialInds)=x2_add{i}(trialInds);
                w2_1(i,c,trialInds)=x3_add{i}(trialInds);
                w2_2(i,c,trialInds)=x4_add{i}(trialInds);
                w(i,c,trialInds) = wealth_add{i}(trialInds);

            case {2}% eta=1
                choice(i,c,trialInds)=choice_mul{i}(trialInds);
                x1_1(i,c,trialInds)=x1_mul{i}(trialInds);
                x1_2(i,c,trialInds)=x2_mul{i}(trialInds);
                x2_1(i,c,trialInds)=x3_mul{i}(trialInds);
                x2_2(i,c,trialInds)=x4_mul{i}(trialInds);
                w(i,c,trialInds) = wealth_current_mul{i}(trialInds);
        end %switch
    end %c
end %i

%% Nan check
disp([num2str(length(find(isnan(choice)))),'_nans in choice data']);      %nans in choice data do not matter
disp([num2str(length(find(isnan(w1_1)))),'_nans in gambles 1_1 matrix'])  %nans in gamble matrices do, since model will not run
disp([num2str(length(find(isnan(w1_2)))),'_nans in gambles 1_2 matrix'])
disp([num2str(length(find(isnan(w2_1)))),'_nans in gambles 2_1 matrix'])
disp([num2str(length(find(isnan(w2_2)))),'_nans in gambles 2_2 matrix'])

%% Configure data structure for graphical model & parameters to monitor
%everything you want jags to use
dataStruct = struct(...
            'nSubjects', nSubjects,'nDynamics',nDynamics,'nTrials',nTrials,...
            'w',w,'w1_1',w1_1,'w1_2',w1_2,'w2_1',w2_1,'w2_2',w2_2,'y',choice,...
            'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muEtaL',muEtaL,'muEtaU',muEtaU,'sigmaEtaL',sigmaEtaL,'sigmaEtaU',sigmaEtaU);

for i = 1:nChains
    monitorParameters = {'beta', 'eta'}
    %monitorParameters = {'y','w','beta','eta','w1_1','w1_2','w2_1','w2_2'}; %To be used for debugging
    S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
end

%% Run JAGS sampling via matJAGS
tic;fprintf( 'Running JAGS ...\n' ); % start clock to time % display

[samples] = matjags( ...
    dataStruct, ...                           % Observed data
    fullfile(jagsDir, [modelName '.txt']), ...% File that contains model definition
    init0, ...                                % Initial values for latent variables
    whichJAGS,...                             % Specifies which copy of JAGS to run on
    'doparallel' , doParallel, ...            % Parallelization flag
    'nchains', nChains,...                    % Number of MCMC chains
    'nburnin', nBurnin,...                    % Number of burnin steps
    'nsamples', nSamples, ...                 % Number of samples to extract
    'thin', nThin, ...                        % Thinning parameter
    'dic', doDIC, ...                         % Do the DIC?
    'monitorparams', monitorParameters, ...   % List of latent variables to monitor
    'savejagsoutput' , 1 , ...                % Save command line output produced by JAGS?
    'verbosity' , 1 , ...                     % 0=do not produce any output; 1=minimal text output; 2=maximum text output
    'cleanup' , 1 ,...                        % clean up of temporary files?
    'rndseed',1);                             % Randomise seed; 0=no; 1=yes

toc %end clock

%% Save stats and samples
disp('saving samples...')
save(['samples_stats/' modelName,'_',data],'samples','-v7.3')

%% Print readouts
%disp('stats:'),disp(stats)%print out structure of stats output
disp('samples:'),disp(samples);%print out structure of samples output
try
    rhats=fields(stats.Rhat);
    for lp = 1: length(rhats)
        disp(['stats.Rhat.',rhats{lp}]);
        eval(strcat('stats.Rhat.',rhats{lp}))
    end
catch
    disp('no field for stats.Rhat')
end
