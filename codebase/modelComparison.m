function modelComparison(data_source, data_dir, model_selection_type)


%data_source = '/1_pilot';
%model_selection_type = 1;

%[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)
%addpath(genpath(fullfile(startDir,'/VBA-toolbox')));

data_poolings = {'no_pooling','partial_pooling','full_pooling'};
%dataDir=fullfile(startDir,'..','/data',data_source);

for ii = 1:length(data_poolings)
    file = sprintf('Bayesian_JAGS_model_selection_%s_%d.mat', data_poolings{ii} ,model_selection_type);
    
    load(fullfile(dataDir, file))
    
    z = samples.z;
    [n_chains, n_samples, n_participants] = size(z);
    z_i = reshape(z, [n_chains * n_samples, n_participants]);
    
    n_models = max(arr(:));
    
    counts = zeros(n, n_participants);
    bin_edges = 1:(n_models+1);
    
    % Loop through each column and count occurrences
    for col = 1:n_participants
        disp(col)
        counts(:, col) = histcounts(z_i(:, col), bin_edges);
    end
    
    counts = counts + 1; % Add 1 to all counts to avoid division by zero
    
    total_counts = sum(counts, 1);
    proportions = counts ./ repmat(total_counts, n, 1);
    
    log_proportions = log10(proportions);
    
    % perform group-BMS on data
    [p1, o1] = VBA_groupBMC (log_proportions, options);
    set (o1.options.handles.hf, 'name', 'group BMS: y_1')
     
end

end