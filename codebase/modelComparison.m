function modelComparison(data_source, model_selection_type)

[startDir,~] = fileparts(mfilename('fullpath'));  %specify your starting directory here (where this script runs from)

dataDir=fullfile(getParentDir(startDir, 1),'/data',data_source);

disp(startDir)
disp(getParentDir(startDir, 1))

switch model_selection_type
    case {1}, name = 'EUT_EE';
    case {2}, name = 'EUT_EE2';
    case {3}, name = 'data_pooling';
end

file = fullfile(dataDir, sprintf('proportions_%s.txt',name));
log_proportions = load(file);
BFFile = sprintf('model_selection_BF_%s.txt', name);

options.verbose = false;
options.DisplayWin = 0;

comparisons = nchoosek(1 : size(log_proportions, 1), 2);
n_comps = size(comparisons, 1);
BFs = zeros(n_comps, 1);

for ii = 1 : n_comps
    BFs(ii) = exp(sum(log_proportions(comparisons(ii, 1), :) - log_proportions(comparisons(ii, 2), :)));
end


[p, o] = VBA_groupBMC(log_proportions, options);

fileID = fopen(fullfile(dataDir, BFFile), 'w');
fprintf(fileID, 'pxp %d\n', o.pxp);
fprintf(fileID, 'ep %d\n', o.ep);
fprintf(fileID, 'Ef %d\n', o.Ef);
fprintf(fileID, 'FFX %d\n', o.Fffx);

for ii = 1 : n_comps
    fprintf(fileID, 'BF_%d_%d %d\n', comparisons(ii, 1), comparisons(ii, 2), BFs(ii));
end

fclose(fileID);

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