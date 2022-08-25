function computeHLM(runModelNum,synthMode,nBurnin,nSamples,nThin,nChains,subjList,whichJAGS,doParallel,startDir)
%% Hiercharchical Latent Mixture (HLM) model
% This is a general script for running several types of hierarchical
% bayesian model via JAGS. It can run hiearchical latent mixture models in
% which different utility models can be compared via inference on the model
% indicator variables, and it can run without latent mixtures of models for
% instance in order to estimate parameters of a given utility model. It
% takes as input the following:

% runModelNum - set which model to run; parameter estimation (1) or model selection (2)
% synthMode   - sets whether to run on real data (1) or synthetic data for parameter recovery (2-3; see setHLM.m for info on differences)
% nBurnin     - specifies how many burn in samples to run
% nSamples    - specifies the number of samples to run
% nThin       - specifies the thinnning number
% nChains     - specifies number of chains
% subjList    - list of subject numbers to include
% whichJAGS   - sets which copy of matjags to run
% doParallel  - sets whether to run chains in parallel
% startDir    - root directory for the repo

%% Set paths
cd(startDir);%move to starting directory
jagsDir=fullfile(startDir,'/HLM_utils/JAGS');
dataDir=fullfile(startDir,'/data');
samplesDir=[startDir,'/data/samples_stats'];

%% Choose & load data
switch synthMode
    case {1}
        data = 'all_data.mat';
        nTrials = 299;
    case {2}
        data = 'synth_data_model_recovery.mat';
        nTrials = 243;
    case {3}
        data = 'synth_data_parameter_recovery'
end

%% Choose JAGS file
switch runModelNum
    case {1}
        modelName = 'JAGS_parameter_recovery';
    case {2,4}
        modelName = 'JAGS_model_selection';
end

%% Set key variables
nTrials=90 #number of trials
nConditions=2;%number of dynamics
doDIC=0;%compute Deviance information criteria? This is the hierarchical equivalent of an AIC, the lower the better
nSubjects=length(subjList);%number of subjects

%% Set bounds of hyperpriors
%hard code the upper and lower bounds of hyperpriors, typically uniformly
%distributed in log space. These values will be imported to JAGS.

%beta - prior on log since cannot be less than 0; note same bounds used for independent priors on all utility models
muLogBetaL=-2.3;muLogBetaU=3.4; %bounds on mean of distribution log beta
sigmaLogBetaL=0.01;sigmaLogBetaU=sqrt(((muLogBetaU-muLogBetaL)^2)/12);%bounds on the std of distribution of log beta

%eta
muEtaL=-2.5;muEtaU=2.5;%bounds on mean of distribution of eta
sigmaEtaL=0.01;sigmaEtaU=sqrt(((muEtaU-muEtaL)^2)/12);%bounds on std of eta

%etaM - prior on log since we are only interested in testing positive change
muLogetaML=-2.3;muLogetaMU=3.4; %bounds on mean of distribution log etaM
sigmaLogetaML=0.01;sigmaLogetaMU=sqrt(((muLogetaMU-muLogetaML)^2)/12);%bounds on the std of distribution of log etaM

%% Print information for user
disp('**************');
disp(['JAGS script: ', modelName])
disp(['on: ', data])
disp(['started: ',datestr(clock)])
disp(['MCMC number: ',num2str(whichJAGS)])
disp('**************');

%% Initialise matrices
%initialise matrices with nan values of size subjects x conditions x trials
dim = nan(nSubjects,nDynamics,nTrials); %specify the dimension
choice = dim; %initialise choice data matrix
g1=dim; g2=dim; g3=dim; g4=dim;%initialise growth-rates
w=dim;%initialise wealth

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
                dx1(i,c,trialInds)=dx1_add{i}(trialInds);%wealth change for outcome 1
                dx2(i,c,trialInds)=dx2_add{i}(trialInds);%same for outcome 2 etc.
                dx3(i,c,trialInds)=dx3_add{i}(trialInds);
                dx4(i,c,trialInds)=dx4_add{i}(trialInds);
                w(i,c,trialInds) = wealth{i}(trialInds);

            case {2}% eta=1
                choice(i,c,trialInds)=choice_mul{i}(trialInds);
                dx1(i,c,trialInds)=dx1_mul{i}(trialInds);%assign changes in wealth dx for outcome 1
                dx2(i,c,trialInds)=dx2_mul{i}(trialInds);%same for outcome 2 etc.
                dx3(i,c,trialInds)=dx3_mul{i}(trialInds);
                dx4(i,c,trialInds)=dx4_mul{i}(trialInds);
                w(i,c,trialInds) = wealth{i}(trialInds);
        end %switch
    end %c
end %i

%% Nan check
disp([num2str(length(find(isnan(choice)))),'_nans in choice data']);%nans in choice data do not matter
disp([num2str(length(find(isnan(dx1)))),'_nans in gambles 1 matrix'])% nans in gamble matrices do, since model will not run
disp([num2str(length(find(isnan(dx2)))),'_nans in gambles 2 matrix'])
disp([num2str(length(find(isnan(dx3)))),'_nans in gambles 3 matrix'])
disp([num2str(length(find(isnan(dx4)))),'_nans in gambles 4 matrix'])

%% Configure data structure for graphical model & parameters to monitor
%everything you want jags to use
switch runModelNum
    case {1}
        dataStruct = struct(...
                    'nSubjects', nSubjects,'nConditions',nConditions,'nTrials',nTrials,...
                    'wealths',w,'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,'y',choice,...
                    'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
                    'muEtaL',muEtaL,'muEtaU',muEtaU,'sigmaEtaL',sigmaEtaL,'sigmaEtaU',sigmaEtaU);

        for i = 1:nChains
            monitorParameters = {'beta','eta'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
        end
    case {2}
        dataStruct = struct(...
                    'nSubjects', nSubjects,'nConditions',nConditions,'nTrials',nTrials,...
                    'wealths',w,'dx1',dx1,'dx2',dx2,'dx3',dx3,'dx4',dx4,'y',choice,...
                    'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
                    'muEtaL',muEtaL,'muEtaU',muEtaU,'sigmaEtaL',sigmaEtaL,'sigmaEtaU',sigmaEtaU,
                    'muEtaML',muEtaML,'muEtaMU',muEtaMU,'sigmaEtaML',sigmaEtaML,'sigmaEtaMU',sigmaEtaMU);

        for i = 1:nChains
            monitorParameters = {'beta','eta','etaM'};
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

toc % end clock

%% Save stats and samples
disp('saving samples...')
save([msamplesDir, modelName,'_',data],'samples','-v7.3')

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