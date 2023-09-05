[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
addpath(genpath(fullfile(startDir,'/VBA-toolbox')));
dataDir=fullfile(startDir,'..','/data','/1_pilot');

load(fullfile(dataDir, 'Bayesian_JAGS_model_selection_partial_pooling.mat'))

z = samples.z;
[n_chains, n_samples, n_participants] = size(z);
z_i = reshape(z, [n_chains * n_samples, n_participants]);

n = max(arr(:));

counts = zeros(n, n_participants);
bin_edges = 1:(n+1);

% Loop through each column and count occurrences
for col = 1:n_participants
    disp(col)
    counts(:, col) = histcounts(z_i(:, col), bin_edges);
end

counts = counts + 1; % Add 1 to all counts to avoid division by zero

total_counts = sum(counts, 1);
proportions = counts ./ repmat(total_counts, n, 1);

log_proportions = log10(proportions);

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
