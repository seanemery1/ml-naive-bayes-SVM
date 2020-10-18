function dataEven =  tableGeneratorEvenSampling(directoryPath)
% Creates a training dataset that is 50% Hole and 50% Not-Hole.

srcLocation =...
    dir(strcat(directoryPath, 'Location\*.mat'));
srcUnlabelled =...
    dir(strcat(directoryPath, 'Unlabelled\*.jpg'));
rng(1)

% Initializing variables
maxIndex = max(size(srcLocation)); % Size of index (150 training images)
count = 0; % Count of pixels, X & Y -> single position on the table
dataEven = zeros(48000000, 6); % Pixel count is an overestimate

% Loop through all 150 unlabbelled images/location matrices
for index = 1:maxIndex
    % Obtaining filenames
    filenameLocation = strcat(directoryPath, 'Location\',...
        srcLocation(index).name);
    filenameUnlabelled = strcat(directoryPath, 'Unlabelled\',...
        srcUnlabelled(index).name);
    
    % Reading files to memory
    location = load(filenameLocation);
    location = location.location;
    cdataUnlabelled = imread(filenameUnlabelled);
    
    % Obtaining image dimensions (n = # of rows, m = # of columns)
    [n, m] = size(location);
    
    % Loop through every pixel in the image
    for height = 1:n
        for width = 1:m
            % If location is a hole or not-hole, mark as hole or not-hole
            % on table
            if location(height,width) == 1 || location(height, width) == -1
                % Iterate through count
                count = count + 1;
                dataEven(count, 1) = height;
                dataEven(count, 2) = width;
                dataEven(count, 3) = cdataUnlabelled(height, width, 1);
                dataEven(count, 4) = cdataUnlabelled(height, width, 2);
                dataEven(count, 5) = cdataUnlabelled(height, width, 3);
                dataEven(count, 6) = location(height,width);
            
            end
        end
    end
end
% Remove any excess, empty rows
dataEven(count+1:end,:) = [];

% Saving Results Matrix
filenameMatrix = sprintf('resultsEven');
nameAndPath = strcat(directoryPath, filenameMatrix); 
save(nameAndPath, 'dataEven');