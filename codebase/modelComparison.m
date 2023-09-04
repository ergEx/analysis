

[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
addpath(genpath(fullfile(startDir,'/VBA-toolbox')));
dataDir=fullfile(startDir,'..','/data','1_pilot');
load(fullfile(dataDir, 'log_proportions.mat'))

% display empirical histogram of log-Bayes factors
% -------------------------------------------------------------------------
plotBayesFactor(log_proportions());


% perform model selection with the VBA
% =========================================================================
options.verbose = false;
options.modelNames = ["EUT", "EE"];

% perform group-BMS on data 
[p1, o1] = VBA_groupBMC (log_proportions, options);
set (o1.options.handles.hf, 'name', 'group BMS: y_1')



%% ########################################################################
% display subfunctions
% #########################################################################
function plotBayesFactor (logEvidence_y1)
    [n1, x1] = VBA_empiricalDensity ((logEvidence_y1(1,:) - logEvidence_y1(2, :))');
    hf = figure ('color' ,'w', 'name', 'demo_modelComparison: distribution of log Bayes factors');
    ha = axes ('parent', hf,'nextplot','add');
    plot (ha, x1, n1, 'color', 'r');
    xlabel (ha, 'log p(y|EUT) - log(y|EE)');
    ylabel (ha, 'proportion of simulations');
end
