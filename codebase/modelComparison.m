function modelComparison(data_source, model_selection_type)

[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)

data_poolings = {'no_pooling','partial_pooling','full_pooling'};
dataDir=fullfile(getParentDir(startDir, 1),'/data',data_source);

disp(startDir)
disp(getParentDir(startDir, 1))


for ii = 1 : length(data_poolings)
    file = sprintf('Bayesian_JAGS_model_selection_%s_%d.mat', data_poolings{ii} ,model_selection_type);
    BFFile = sprintf('model_selection_BF_%s_%d.txt', data_poolings{ii} ,model_selection_type);

    jags_dat = load(fullfile(dataDir, file));

    z = jags_dat.samples.z;
    [n_chains, n_samples, n_participants] = size(z);
    z_i = reshape(z, [n_chains * n_samples, n_participants]);
    z_i = mod(z_i, 2) + 1; %note this changes the order such that m1=2, m2=1

    n_models = max(z_i(:));

    counts = zeros(n_models, n_participants);
    bin_edges = 1:(n_models+1);

    % Loop through each column and count occurrences
    for col = 1:n_participants
        counts(:, col) = histcounts(z_i(:, col), bin_edges);
    end

    counts = counts + 1; % Add 1 to all counts to avoid division by zero

    total_counts = sum(counts, 1);
    proportions = counts ./ repmat(total_counts, n_models, 1);
    proportions_file = sprintf('proportions_%s_%d.csv', data_poolings{ii}, model_selection_type);

    log_proportions = log10(proportions);

    switch model_selection_type
        case {1}, VariableNames = {'EE', 'EUT'};
        case {2}, VariableNames = {'Weak EE', 'EUT'}
    end

    writetable(array2table(proportions', 'VariableNames', VariableNames), fullfile(dataDir, proportions_file));

    BF = exp(sum(log_proportions(1, :)) - sum(log_proportions(2, :)));

    fileID = fopen(fullfile(dataDir, BFFile), 'w');
    fprintf(fileID, ['BF', num2str(BF)]);
    fclose(fileID);

end

end

function newDir = getParentDir(dir,numUpDirs)
    %   Function to get parent dir from either a file or a directory, going up
    %   the number of directories indicated by numUpDirs.
    %
    %   dir = string (filepath or pwd)
    %   numUpDirs = positive integer
    %
    %   Written by: Walter Adame Gonzalez
    %   McGill University
    %   walter.adamegonzalez@mail.mcgill.ca
    %   slightly updated by SRSteinkamp

    if nargin < 2
        numUpDirs = 1;
    end


    pre = '';
    if ispc

        if dir(1) == '\'
            pre = '\';
        end
        parts = strsplit(dir, '\');

    else
        if dir(1) == '/'
            pre = '/';
        end
        parts = strsplit(dir, '/');
    end

    newDir = '';
    if numUpDirs<length(parts)
        for i=1:(length(parts)-numUpDirs)
        newDir = fullfile(newDir,string(parts(i)));
        end
    else
        disp("numUpDirs indicated is larger than the number of possible parent directories. Returning the unchanged dir")
        newDir = dir;
    end
    newDir = [char(pre) char(newDir)];
end