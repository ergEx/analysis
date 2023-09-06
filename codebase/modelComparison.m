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
    BFFile = sprintf('model_selection_BF_%s_%d.txt', data_poolings{ii} ,model_selection_type);
    
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
    
    BF10 = exp(sum(log_proportions(2, :)) - sum(log_proportions(1, :)));
    fprintf(num2str(sum(log_proportions(2, :))))
    fprintf(num2str(sum(log_proportions(1, :))))
    fprintf(num2str(exp(sum(log_proportions(2, :)))))
    fprintf(num2str(exp(sum(log_proportions(1, :)))))
    BF = ['BF10 ', num2str(BF10)];
    
    if n_models == 3
        BF20 = exp(sum(log_proportions(3, :)) - sum(log_proportions(1, :)));
        BF21 = exp(sum(log_proportions(3, :)) - sum(log_proportions(2, :)));
        BF = [BF, '\n', 'BF20 ', num2str(BF20), '\n', 'BF21 ', num2str(BF21)];
    end
    
    if exist(fullfile(dataDir, BFFile), 'file')
        fileID = fopen(fullfile(dataDir, BFFile), 'w');
        fprintf(fileID, BF);
        fclose(fileID);
    else
        fileID = fopen(fullfile(dataDir, BFFile), 'w');
        fprintf(fileID, BF);
        fclose(fileID);
    end
    
    
    options = {};
    % perform group-BMS on data
    [p1, o1] = VBA_groupBMC (log_proportions, options);
    set (o1.options.handles.hf, 'name', 'group BMS: y_1')
    saveas(gcf, fullfile(figDir, sprintf('model_selection_%s_%i.pdf', data_poolings{ii}, model_selection_type)));
end

end