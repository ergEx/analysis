function computeBayesian(~,dataPooling,inferenceMode,model_selection_type,nBurnin,nSamples,nThin,nChains,subjList,whichJAGS,doParallel,startDir,nTrials,folder,seedChoice)
%% Hiercharchical Latent Mixture (HLM) model
% This is a general script for running several types of hierarchical
% bayesian model via JAGS. It can run hiearchical latent mixture models in
% which different utility models can be compared via inference on the model
% indicator variables, and it can run without latent mixtures of models for
% instance in order to estimate parameters of a given utility model. It
% takes as input the following:

% DataSource  - set which data source is used; Simualtion (0)
%                                              Pilot (1)
%                                              Full experiment (2)
% dataPooling - set whether to do No pooling (1)
%                                 Partial pooling (individual estimates from group level distributions) (2)
%                                 Full pooling (super individual) (3)
% inferenceMode - set whether to do parameter estimation (1)
%                                   Bayesian model comparison of three different models (2)
%                                   Bayesian model comparison of data pooling (2)
% whichJAGS     - which copy of matjags to run on. this allows parallel jobs to run as long as they use different matjags
% nBurnin       - specifies how many burn in samples to run
% nSamples      - specifies the number of samples to run
% nThin         - specifies the thinnning number
% nChains       - specifies number of chains
% subjList      - list of subject numbers to include
% whichJAGS     - sets which copy of matjags to run
% doParallel    - sets whether to run chains in parallel
% startDir      - root directory for the repo
% nTrials       - number of trials in experiment
% folder        - folder within the datafolder the relevant data is stored

%% Set paths
disp(startDir)
cd(startDir);%move to starting directory
matjagsdir=fullfile(startDir,'/Bayesian_utils/matjags');
addpath(matjagsdir)
jagsDir=fullfile(startDir,'/Bayesian_utils/JAGS');
addpath(jagsDir)
dataDir=fullfile('..','data',folder);

%% Choose & load data
load(fullfile(dataDir, 'all_active_phase_data.mat'))

%% Choose JAGS file
switch inferenceMode
    case {1}
        switch dataPooling
            case {1}, modelName = 'JAGS_parameter_estimation_no_pooling';
            case {2}, modelName = 'JAGS_parameter_estimation_partial_pooling';
            case {3}, modelName = 'JAGS_parameter_estimation_full_pooling';
        end %switch dataPooling
    case {2}
        switch dataPooling
            case {1}, modelName = 'JAGS_model_selection_no_pooling';
            case {2}, modelName = 'JAGS_model_selection_partial_pooling';
            case {3}, modelName = 'JAGS_model_selection_full_pooling';
        end %switch dataPooling
    case {3}, modelName = 'JAGS_model_selection_data_pooling'
end %switch inferenceMode

%% Set key variables
nConditions=2;%number of dynamics
doDIC=0;%compute Deviance information criteria? This is the hierarchical equivalent of an AIC, the lower the better
nSubjects=length(subjList);%number of subjects

%% Set bounds of hyperpriors
%hard code the upper and lower bounds of hyperpriors, typically uniformly
%distributed in log space. These values will be imported to JAGS.

sigma_l = 0.01;
sigma_h =1.61;
mu_eta_l = -4;
mu_eta_h = 5;
d_mu_alpha = 2;
d_mu_beta = 0.5;
mu_log_beta_l = -2.3;
mu_log_beta_h = 3.4;
mu_eta_EE_add = 0.0000;
mu_eta_EE_mul = 0.9999;

%Model indicator
switch inferenceMode
    case{1}
        %no model indicator used for parameter estimation
    case {2}
        switch model_selection_type
            case {1}, pz = [1/2, 1/2, 0];   %only test model 1 v model 2
            case {2}, pz = [1/3, 1/3, 1/3]; %flat prior over all three models
        end
    case {3}
        pz = [1/3, 1/3, 1/3]; %flat prior over all three data pooling methods
end


%% Print information for user
disp('**************');
disp(['Mode: ', modelName])
disp(['dataSource: ', folder])
disp(['started: ',datestr(clock)])
disp(['MCMC number: ',num2str(whichJAGS)])
disp(['Subs: ' num2str(nSubjects), ' Conds: ' num2str(nConditions), ' Trials : ' num2str(nTrials)])
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
    case {1}
        dataStruct = struct(...
            'nSubjects', nSubjects,'nConditions',nConditions,'nTrials',nTrials,...
            'w',w,'dwLU',dwLU,'dwLL',dwLL,'dwRU',dwRU,'dwRL',dwRL,'y',choice,...
            'sigmaL',sigma_l,'sigmaH',sigma_h,...
            'muEtaL',mu_eta_l,'muEtaH',mu_eta_h,...
            'muLogBetaL',mu_log_beta_l,'muLogBetaH',mu_log_beta_h);
        
        for i = 1:nChains
            monitorParameters = {'mu_eta','tau_eta','sigma_eta',...
                'mu_log_beta','tau_log_beta','sigma_log_beta',...
                'beta_i', 'beta_g','eta_i', 'eta_g'};
            S=struct; init0(i)=S;
        end %i
    case {2}
        dataStruct = struct(...
            'nSubjects', nSubjects,'nConditions',nConditions,'nTrials',nTrials,...
            'w',w,'dwLU',dwLU,'dwLL',dwLL,'dwRU',dwRU,'dwRL',dwRL,'y',choice,...
            'sigmaL',sigma_l,'sigmaH',sigma_h,...
            'muEtaL',mu_eta_l,'muEtaH',mu_eta_h,...
            'muEtaEEAdd',mu_eta_EE_add,'muEtaEEMul',mu_eta_EE_mul,...
            'dMuAlpha',d_mu_alpha,'dMuBeta',d_mu_beta,...
            'muLogBetaL',mu_log_beta_l,'muLogBetaH',mu_log_beta_h,...
            'pz',pz);
        
        for i = 1:nChains
            monitorParameters = {'beta_i_EUT', 'beta_i_EE','beta_i_EE2',...
                'eta_i_EUT', 'eta_i_EE', 'eta_i_EE2',...
                'z'};
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
        end %i
    case {3}
        dataStruct = struct(...
            'nSubjects', nSubjects,'nConditions',nConditions,'nTrials',nTrials,...
            'w',w,'dwLU',dwLU,'dwLL',dwLL,'dwRU',dwRU,'dwRL',dwRL,'y',choice,...
            'sigmaL',sigma_l,'sigmaH',sigma_h,...
            'muEtaL',mu_eta_l,'muEtaH',mu_eta_h,...
            'muLogBetaL',mu_log_beta_l,'muLogBetaH',mu_log_beta_h,...
            'pz',pz);
        
        for i = 1:nChains
            monitorParameters = {'beta_i_1', 'beta_g_1','eta_i_1', 'eta_g_1',... %no pooling
                'beta_i_2', 'beta_g_2','eta_i_2', 'eta_g_2',... %partial pooling
                'beta_i_3', 'beta_g_3','eta_i_3', 'eta_g_3',... %full pooling
                'z'};%model indicator
            S=struct; init0(i)=S; %sets initial values as empty so randomly seeded
        end %i
end %switch
%% Run JAGS sampling via matJAGS
tic;fprintf( 'Running JAGS ...\n' ); % start clock to time % display

[samples, stats] = matjags( ...
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
    'rndseed',seedChoice);                    % Randomise seed; 1=no; 2=yes

toc % end clock

%% Save stats and samples
disp('saving samples...')

if inferenceMode == 2
    save(fullfile(dataDir, append('Bayesian','_',modelName,'_',num2str(model_selection_type))),'stats','samples','-v7.3')
else
    save(fullfile(dataDir, append('Bayesian','_',modelName)),'stats','samples','-v7.3')
end
%% Print readouts
disp('stats:'),disp(stats)%print out structure of stats output
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
