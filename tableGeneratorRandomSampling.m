function dataRandom = tableGeneratorRandomSampling(directoryPath)
% Creates a representative training dataset that is a downsample of the 
% original 150 images, approximately 99% Hole and 1% Not-Hole.

srcLocation =...
    dir(strcat(directoryPath, 'Location\*.mat'));
srcUnlabelled =...
    dir(strcat(directoryPath, 'Unlabelled\*.jpg'));

% Initializing variables
maxIndex = max(size(srcLocation)); % Size of index (150 training images)
count = 0; % Count of pixels, X & Y -> single position on the table
dataRandom = zeros(48000000, 6); % Pixel count is an overestimate
    
% Loop through all 150 unlabelled images/location matrices
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
            % Generating a random integer between 1 and 15 (inclusive)
            rand = randi([1 15]);
            % If random integer = 1, sample the pixel
            if rand == 1
                % Iterate through count
                count = count + 1;
                if location(height,width) == 1
                    dataRandom(count, 1) = height;
                    dataRandom(count, 2) = width;
                    dataRandom(count, 3) =...
                        cdataUnlabelled(height, width, 1);
                    dataRandom(count, 4) =...
                        cdataUnlabelled(height, width, 2);
                    dataRandom(count, 5) =...
                        cdataUnlabelled(height, width, 3);
                    dataRandom(count, 6) = 1;
                else
                    dataRandom(count, 1) = height;
                    dataRandom(count, 2) = width;
                    dataRandom(count, 3) =...
                        cdataUnlabelled(height, width, 1);
                    dataRandom(count, 4) =...
                        cdataUnlabelled(height, width, 2);
                    dataRandom(count, 5) =...
                    cdataUnlabelled(height, width, 3);
                    dataRandom(count, 6) = -1;
                end
            end
        end
    end
end
% Remove any excess, empty rows
dataRandom(count+1:end,:) = [];

% Saving Results Matrix
filenameMatrix = sprintf('resultsRandom');
nameAndPath = strcat(directoryPath, filenameMatrix); 
save(nameAndPath, 'dataRandom');
end
    
    
