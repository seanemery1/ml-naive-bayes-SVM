function imagePreProcessing(directoryPath)
% Creates Locatin matrices for all 150 training images, while also prepping
% the tableGenerator functions to generate "Randomly Sampled" data or
% "Evenly Sampled data."

srcLabelled =...
    dir(strcat(directoryPath, 'Labelled\*.png'));
srcUnlabelled =...
    dir(strcat(directoryPath, 'Unlabelled\*.jpg'));

count = 0;
indexMax = max(size(srcLabelled));

% Loop through all 150 unlabbelled images/location matrices
for index = 1:indexMax
    filenameLabelled = strcat(directoryPath, 'Labelled\',...
        srcLabelled(index).name);
    filenameUnlabelled = strcat(directoryPath, 'Unlabelled\',...
        srcUnlabelled(index).name);
    
    % Reading filenames as images
    cdataLabelled = imread(filenameLabelled);
    cdataUnlabelled = imread(filenameUnlabelled);
    
    % Converting labelled images to grayscale (grayscale 84 = hole)
    labelledGray = rgb2gray(cdataLabelled);
    
    % Obtaining dimensions of images
    [n, m] = size(labelledGray);
    
    % Loop through every pixel in the image
    for height = 1:n
        for width = 1:m
            % If pixel is not hole, paint black
            if labelledGray(height,width) ~= 84
                labelledGray(height,width) = 0;
            % Else if pixel is hole, paint white
            else
                labelledGray(height,width) = 255;
            end
        end
    end
    
    % Converting Noisy Image to Black and White
    blackWhiteNoisy = imbinarize(labelledGray);
    
    % Saving Noisy Image
    filenameMatrix = sprintf(srcUnlabelled(index).name);
    filenamePNGified = strrep(filenameMatrix,'jpg','png');
    nameAndPath = strcat(directoryPath, 'blackWhiteNoisy\',...
        filenamePNGified); 
    imwrite(blackWhiteNoisy, nameAndPath);
    
    % Removing Noise
    labelledBlackWhite = bwareaopen(labelledGray, 250);
    props = regionprops(labelledBlackWhite, 'Area', 'BoundingBox');

    % Saving No Noise Image (Random Sampling)
    filenameMatrix = sprintf(srcUnlabelled(index).name);
    filenamePNGified = strrep(filenameMatrix,'jpg','png');
    nameAndPath = strcat(directoryPath, 'blackWhiteRandomSampling\',...
        filenamePNGified);
    imwrite(labelledBlackWhite, nameAndPath);
    
    % Creating a Bounded Box for Even Sampling
    left = floor(props.BoundingBox(1));
    top = floor(props.BoundingBox(2));
    right = floor(props.BoundingBox(1)) + props.BoundingBox(3) + 1;
    bottom = floor(props.BoundingBox(2)) + props.BoundingBox(4) + 1;
    xrange = props.BoundingBox(3) + 1;
    yrange = props.BoundingBox(4) + 1;
    
    % Randomizing Not-Hole Sampling Location
    randX = randi([1 (m - xrange - 1)]);
    randY = randi([1 (n - yrange - 1)]);

    % If Not-Hole is within range of bounded box of hole, re-randomize
    while ((left - xrange < randX && randX < right)...
            && (top - yrange < randY &&  randY < bottom))
        randX = randi([1 (m - xrange - 1)]);
        randY = randi([1 (n - yrange - 1)]);
    end

    % Initializing matrix of size n, m to store labels of 1/-1
    location1 = int8(zeros(n,m));
    location2 = int8(zeros(n,m));
    
    % Loop through black/white image within the bounded box to draw a
    % circle at a random location on the image.
    for height = top:bottom
        for width = left:right
            if labelledBlackWhite(height, width) == 1
                % Storing hole location
                location1(height, width) = 1;
                % Storing not hole location
                location2(height - top - 1 + randY,...
                    width - left - 1 + randX) = -1;
                % Storing black and white image reference
                labelledBlackWhite(height - top - 1 + randY,...
                    width - left - 1 + randX) = 1;
            end
        end       
    end
    
    % Initializing black and white visual reference/aid
    blackWhiteVisualReference = ones(n, m);
    
    % Creating a visual reference of output (Unnecessary, but useful to see
    % if we did things correctly)
    for height = 1:n
        for width = 1:m
            if labelledBlackWhite(height, width) == 0
                count = count+1;
                cdataUnlabelled(height, width, 1) = 255;
                cdataUnlabelled(height, width, 2) = 255;
                cdataUnlabelled(height, width, 3) = 255;
                blackWhiteVisualReference(height, width) = 0;
            end
        end       
    end
    
    % Saving the location matrix as a black and white image (for visual
    % reference)
    
    filenameMatrix = sprintf(srcUnlabelled(index).name);
    filenamePNGified = strrep(filenameMatrix,'jpg','png');
    nameAndPath = strcat(directoryPath, 'BlackWhiteEven\',...
        filenamePNGified); 
    imwrite(blackWhiteVisualReference, nameAndPath);
    
    % Combine hole with randomly sampled non-hole area
    location = location1 + location2;
 
    % Saving the location matrix
    filenameMatrix = sprintf(srcUnlabelled(index).name);
    filenameMATified = strrep(filenameMatrix,'jpg','mat');
    nameAndPath = strcat(directoryPath, 'Location\', filenameMATified); 
    save(nameAndPath, 'location');
    
    % Saving visual reference of location matrices
    filenameMatrix = sprintf(srcUnlabelled(index).name);
    filenamePNGified = strrep(filenameMatrix,'jpg','png');
    nameAndPath = strcat(directoryPath, 'HoleEvenSampling\',...
        filenamePNGified); 
    imwrite(cdataUnlabelled, nameAndPath);
end
end