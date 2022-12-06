function computeHLM(inferenceMode,synthMode,aggregationMode,nBurnin,nSamples,nThin,nChains,subjList,whichJAGS,doParallel,startDir,dataVersion,nTrials)
%% Hiercharchical Latent Mixture (HLM) model
% This is a general script for running several types of hierarchical
% bayesian model via JAGS. It can run hiearchical latent mixture models in
% which different utility models can be compared via inference on the model
% indicator variables, and it can run without latent mixtures of models for
% instance in order to estimate parameters of a given utility model. It
% takes as input the following:

% inferenceMode - set whether to do patameter estimation (1) or model selection (2)
% synthMode     - sets how to simulate data ; (1) Real data
%                                           (2) Simulated data
% nBurnin       - specifies how many burn in samples to run
% nSamples      - specifies the number of samples to run
% nThin         - specifies the thinnning number
% nChains       - specifies number of chains
% subjList      - list of subject numbers to include
% whichJAGS     - sets which copy of matjags to run
% doParallel    - sets whether to run chains in parallel
% startDir      - root directory for the repo
% version       - which version of experimental setup to run on;(1) One gamble version
%                                                               (2) Two gamble version
%                                                               (3) Two gamble version w. wealth controls
%                                                               (4) Two gamble version w. different additive c
%                                                               (5) Two gamble version w. hidden wealth

%% Set paths
cd(startDir);%move to starting directory
matjagsdir=fullfile(startDir,'/HLM_utils/matjags');
addpath(matjagsdir)
jagsDir=fullfile(startDir,'/HLM_utils/JAGS');
addpath(jagsDir)
dataDir=fullfile(startDir,'/data',dataVersion);
simulationDir=fullfile(startDir,'/data',dataVersion,'simulations');

%% Choose & load data
mode = 'estimate data';
switch synthMode
    case {1}
        dataSource = 'real_data';
        load(fullfile(dataDir, 'all_active_phase_data.mat'))
    case {2}
        dataSource = 'simulated_data';
        subjList = 1:2;
        nTrials = 160;
        load(fullfile(simulationDir,'all_active_phase_data'))
end %switch synthMode

%% Choose JAGS file
switch inferenceMode
    case {1}, modelName = 'JAGS_parameter_estimation'; mode = 'parameter_estimation';
    case {2}, modelName = 'JAGS_model_selection'; mode = 'model_selection';
end %switch inferencemode

%% Set key variables
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


%% Print information for user
disp('**************');
disp(['Mode: ', mode])
disp(['Data version: ', dataVersion])
disp(['dataSource: ', dataSource])
disp(['started: ',datestr(clock)])
disp(['MCMC number: ',num2str(whichJAGS)])
disp('**************');

%% Initialise matrices
%initialise matrices with nan values of size subjects x conditions x trials
dim = nan(nSubjects,nConditions,nTrials); %specify the dimension
choice = dim; %initialise choice data matrix
dwLU=dim; dwLL=dim; dwRU=dim; dwRL=dim;%initialise growth-rates
w=dim;%initialise wealth

%% Compile choice & gamble data
% Jags cannot deal with partial observations, so we need to specify gamble info for all nodes. This doesn't change anything.

%We allow all agents to have different session length. Therefore we add random gambles to pad out to maxTrials. This
%allows jags to work since doesn't work for partial observation. This does not affect
%parameter estimation. nans in the choice data are allowed as long as all covariates are not nan.

%shift wealth data
w_dim = size(wealth_add);
start_w = 1000.*ones(w_dim(1),1);
wealth_add = [start_w, wealth_add];
wealth_add = wealth_add(:,1:end-1);
wealth_mul = [start_w, wealth_add];
wealth_mul = wealth_mul(:,1:end-1);

%flatten variables if aggregating
if aggregationMode == 2
    choice_add = permute(reshape(choice_add, [], nSubjects),[2,1]);
    x1_1_add   = permute(reshape(x1_1_add, [], nSubjects),[2,1]);
    x1_2_add   = permute(reshape(x1_2_add, [], nSubjects),[2,1]);
    x2_1_add   = permute(reshape(x2_1_add, [], nSubjects),[2,1]);
    x2_2_add   = permute(reshape(x2_2_add, [], nSubjects),[2,1]);
    wealth_add = permute(reshape(wealth_add, [], nSubjects),[2,1]);

    choice_mul = permute(reshape(choice_mul, [], nSubjects),[2,1]);
    x1_1_mul   = permute(reshape(x1_1_mul, [], nSubjects),[2,1]);
    x1_2_mul   = permute(reshape(x1_2_mul, [], nSubjects),[2,1]);
    x2_1_mul   = permute(reshape(x2_1_mul, [], nSubjects),[2,1]);
    x2_2_mul   = permute(reshape(x2_2_mul, [], nSubjects),[2,1]);
    wealth_mul = permute(reshape(wealth_mul, [], nSubjects),[2,1]);

end %if



trialInds = 1:nTrials;
for c = 1:nConditions
    switch c
        case {1} %eta = 0
            choice(:,c,trialInds)=choice_add(:,trialInds);
            dwLU(:,c,trialInds)=x1_1_add(:,trialInds);
            dwLL(:,c,trialInds)=x1_2_add(:,trialInds);
            dwRU(:,c,trialInds)=x2_1_add(:,trialInds);
            dwRL(:,c,trialInds)=x2_2_add(:,trialInds);
            w(:,c,trialInds)=wealth_add(:,trialInds);

        case {2}% eta=1
            choice(:,c,trialInds)=choice_mul(:,trialInds);
            dwLU(:,c,trialInds)=x1_1_mul(:,trialInds);
            dwLL(:,c,trialInds)=x1_2_mul(:,trialInds);
            dwRU(:,c,trialInds)=x2_1_mul(:,trialInds);
            dwRL(:,c,trialInds)=x2_2_mul(:,trialInds);
            w(:,c,trialInds)=wealth_mul(:,trialInds);
    end %switch
end %c



%% Nan check
disp([num2str(length(find(isnan(choice)))),'_nans in choice data']);%nans in choice data do not matter
disp([num2str(length(find(isnan(dwLU)))),'_nans in gambles Left Upper matrix'])% nans in gamble matrices do
disp([num2str(length(find(isnan(dwLL)))),'_nans in gambles Left Lower matrix'])
disp([num2str(length(find(isnan(dwRU)))),'_nans in gambles Right Upper matrix'])
disp([num2str(length(find(isnan(dwRL)))),'_nans in gambles Right Lower matrix'])
disp([num2str(length(find(isnan(w)))),'_nans in wealth matrix'])

%% Configure data structure for graphical model & parameters to monitor
%everything you want jags to use
switch inferenceMode
    case {1} %estimate parameters
        dataStruct = struct(...
                    'nSubjects', nSubjects,'nConditions',nConditions,'nTrials',nTrials,...
                    'w',w,'dwLU',dwLU,'dwLL',dwLL,'dwRU',dwRU,'dwRL',dwRL,'y',choice,...
                    'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
                    'muEtaL',muEtaL,'muEtaU',muEtaU,'sigmaEtaL',sigmaEtaL,'sigmaEtaU',sigmaEtaU);

        for i = 1:nChains
            monitorParameters = {'mu_eta','tau_eta','sigma_eta',...
                                 'mu_log_beta','tau_log_beta','sigma_log_beta',...
                                 'log_beta','beta','eta'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
        end %i
    case {2} %model selection

end %switch inferenceMode

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
    'cleanup' , 0 ,...                        % clean up of temporary files?
    'rndseed',1);                             % Randomise seed; 0=no; 1=yes

toc % end clock

%% Save stats and samples
disp('saving samples...')
save(fullfile(dataDir, append(mode,'_',dataSource,int2str(nSamples))),'samples','-v7.3')

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
