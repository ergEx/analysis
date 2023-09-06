function modelComparison(data_source, model_selection_type)


%data_source = '/1_pilot';
%model_selection_type = 1;

[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
%addpath(genpath(fullfile(startDir,'/VBA-toolbox')));

base = pwd;
[vba_dir, ~ ] = fileparts(which('VBA_setup'));
cd(vba_dir)
VBA_setup;
cd(base)

data_poolings = {'no_pooling','partial_pooling','full_pooling'};
dataDir=fullfile(startDir,'..','/data',data_source);
figDir = fullfile(startDir, '..', '/figs', data_source)

for ii = 1:length(data_poolings)
    file = sprintf('Bayesian_JAGS_model_selection_%s_%d.mat', data_poolings{ii} ,model_selection_type);
    
    jags_dat = load(fullfile(dataDir, file))
    
    z = jags_dat.samples.z;
    [n_chains, n_samples, n_participants] = size(z);
    z_i = reshape(z, [n_chains * n_samples, n_participants]);
    
    n_models = max(z(:));
    
    counts = zeros(n_models, n_participants);
    bin_edges = 1:(n_models+1);
    
    % Loop through each column and count occurrences
    for col = 1:n_participants
        disp(col)
        counts(:, col) = histcounts(z_i(:, col), bin_edges);
    end
    
    counts = counts + 1; % Add 1 to all counts to avoid division by zero
    
    total_counts = sum(counts, 1);
    proportions = counts ./ repmat(total_counts, n_models, 1);
    
    log_proportions = log10(proportions);
    options = {};
    % perform group-BMS on data
    [p1, o1] = VBA_groupBMC (log_proportions, options);
    set (o1.options.handles.hf, 'name', 'group BMS: y_1')
    saveas(gcf, fullfile(figDir, sprintf('model_selection_%s_%i.pdf', ii, model_selection_type)));
end

end