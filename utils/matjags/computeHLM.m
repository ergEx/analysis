function computeHLM(runModelNum,nBurnin,nSamples,nThin,nChains,subjList,whichJAGS,doParallel)
%% Hiercharchical Latent Mixture (HLM) model
% This is a general script for running several types of hierarchical
% bayesian model via JAGS. It can run hiearchical latent mixture models in
% which different utility models can be compared via inference on the model
% indicator variables, and it can run without latent mixtures of models for
% instance in order to estimate parameters of a given utility model. It
% takes as input the following:

% runModelNum - a number which selects which JAGS model to run.
% nBurnin     - a number which specifies how many burn in samples to run.
% nSamples    - a number specifying the number of samples to run
% nThin       - a number specifying the thinnning number
% nChains     - number of chains
% subjList    - list of subject numbers to include
% whichJAGS   - sets which copy of matjags to run
% runPlots    - sets whether to plot data/illustrations(1) or to suppress (0)
% synthMode   - sets whether to simulate data (0), run on real data (1), synthetic data for parameter recovery (2-10)
% doParallel  - sets whether to run chains in parallel

%% Set paths
[startDir,~] = fileparts(mfilename('fullpath'));%specify your starting directory here (where this script runs from)
cd(startDir);%move to starting directory
jagsDir=fullfile(startDir,'/JAGS');
dataDir=fullfile(startDir,'..','/data');
samplesDir=[startDir,'/samples_stats'];

%% Choose & load data 
switch runModelNum
    case {1,2}
        data = 'CPH_data_all_data.mat';
        load(fullfile(dataDir,data))
        nTrials = 299;
    case {3,4}
        data = 'CPH_data_optimal_eta_trials_deleted.mat'; 
        load(fullfile(dataDir,data))
        nTrials = 243;
end
%% Choose JAGS file
switch runModelNum
    case {1,3}
        modelName = 'JAGS_constant_starting_wealth';
    case {2,4}
        modelName = 'JAGS_current_wealth';
end
%% Set key variables
nDynamics=2;%number of dynamics
doDIC=0;%compute Deviance information criteria? This is the hierarchical equivalent of an AIC, the lower the better
nSubjects=length(subjList);%number of agents

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
disp(['on: ', data])
disp(['started: ',datestr(clock)])
disp(['MCMC number: ',num2str(whichJAGS)])
disp('**************');

%% Initialise matrices
%initialise matrices with nan values of size subjects x conditions x trials
dim = nan(nSubjects,nDynamics,nTrials); %specify the dimension
choice = dim; %initialise choice data matrix 
g1=dim; g2=dim; g3=dim; g4=dim;%initialise growth-rates

%% Compile choice & gamble data
% Jags cannot deal with partial observations, so we need to specify gamble info for all nodes. This doesn't change anything.

%We allow all agents to have different session length. Therefore we add random gambles to pad out to maxTrials. This
%allows jags to work since doesn't work for partial observation. This does not affect
%parameter estimation. nans in the choice data are allowed as long as all covariates are not nan.
switch runModelNum
    case {1,3} %starting wealth
        w = nan(nSubjects,nDynamics);%initialise wealth
        for i = 1:nSubjects
            for c = 1:nDynamics
                trialInds=1:nTrials;
                switch c
                    case {1} %eta = 0
                        choice(i,c,trialInds)=choice_add{i}(trialInds);
                        g1(i,c,trialInds)=x1_add{i}(trialInds);%assign growth rate for outcome 1
                        g2(i,c,trialInds)=x2_add{i}(trialInds);%same for outcome 2 etc.
                        g3(i,c,trialInds)=x3_add{i}(trialInds);
                        g4(i,c,trialInds)=x4_add{i}(trialInds);
                        w(i,c) = wealth_add(i);
                    
                    case {2}% eta=1
                        choice(i,c,trialInds)=choice_mul{i}(trialInds);
                        g1(i,c,trialInds)=x1_mul{i}(trialInds);%assign changes in wealth dx for outcome 1
                        g2(i,c,trialInds)=x2_mul{i}(trialInds);%same for outcome 2 etc.
                        g3(i,c,trialInds)=x3_mul{i}(trialInds);
                        g4(i,c,trialInds)=x4_mul{i}(trialInds);
                        w(i,c) = wealth_mul(i);
                end %switch
            end %c
        end %i
    case {2,4} %updating wealth
        w = nan(nSubjects,nDynamics,nTrials);%initialise wealth
        for i = 1:nSubjects
            for c = 1:nDynamics
                trialInds=1:nTrials;
                switch c
                    case {1} %eta = 0
                        choice(i,c,trialInds)=choice_add{i}(trialInds);
                        g1(i,c,trialInds)=dx1_add{i}(trialInds);%assign growth rate for outcome 1
                        g2(i,c,trialInds)=dx2_add{i}(trialInds);%same for outcome 2 etc.
                        g3(i,c,trialInds)=dx3_add{i}(trialInds);
                        g4(i,c,trialInds)=dx4_add{i}(trialInds);
                        w(i,c,trialInds) = wealth_current_add{i}(trialInds);
                    
                    case {2}% eta=1
                        choice(i,c,trialInds)=choice_mul{i}(trialInds);
                        g1(i,c,trialInds)=dx1_mul{i}(trialInds);%assign changes in wealth dx for outcome 1
                        g2(i,c,trialInds)=dx2_mul{i}(trialInds);%same for outcome 2 etc.
                        g3(i,c,trialInds)=dx3_mul{i}(trialInds);
                        g4(i,c,trialInds)=dx4_mul{i}(trialInds);
                        w(i,c,trialInds) = wealth_current_mul{i}(trialInds);
                end %switch
            end %c
        end %i
end %switch

%% Nan check
disp([num2str(length(find(isnan(choice)))),'_nans in choice data']);%nans in choice data do not matter
disp([num2str(length(find(isnan(g1)))),'_nans in gambles 1 matrix'])% nans in gamble matrices do, since model will not run
disp([num2str(length(find(isnan(g2)))),'_nans in gambles 2 matrix'])
disp([num2str(length(find(isnan(g3)))),'_nans in gambles 3 matrix'])
disp([num2str(length(find(isnan(g4)))),'_nans in gambles 4 matrix'])

%% Configure data structure for graphical model & parameters to monitor
%everything you want jags to use
dataStruct = struct(...
            'nSubjects', nSubjects,'nConditions',nDynamics,'nTrials',nTrials,...
            'wealths',w,'g1',g1,'g2',g2,'g3',g3,'g4',g4,'y',choice,...
            'muLogBetaL',muLogBetaL,'muLogBetaU',muLogBetaU,'sigmaLogBetaL',sigmaLogBetaL,'sigmaLogBetaU',sigmaLogBetaU,...
            'muEtaL',muEtaL,'muEtaU',muEtaU,'sigmaEtaL',sigmaEtaL,'sigmaEtaU',sigmaEtaU);

for i = 1:nChains
    monitorParameters = {'y','wealths','beta','eta','g1','g2','g3','g4'};
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
