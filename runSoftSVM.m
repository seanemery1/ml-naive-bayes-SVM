function [confusionMatrixAvg, bestImageName, confusionMatrixBest,...
    worstImageName, confusionMatrixWorst] =...
    runSoftSVM(X, Y, directoryPath, lambda, maxIter, trickSTD, evenOrRandom)
% Runs custom SVM based off of PEGASOS. Optional parameter of
% standardization .

% Initializing directory names
srcTestLabelled =...
    dir(strcat(directoryPath, 'testLabelled\*.jpg'));
srcTestUnlabelled =...
    dir(strcat(directoryPath, 'testUnlabelled\*.jpg'));

% Initializing max index as number of images in testLabelled/testUnlabelled
maxIndex = max(size(srcTestUnlabelled)); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Stochastic Sub-Gradient Descent for Soft Margin SVM Code %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initializing weight as 0
[n, m] = size(X);
w = zeros(1, m);

% If trickSTD = true, standardize results and square standardized
% height/width (height/width is closest to 0 for the hole)
if trickSTD == true
    X = X'; % Transpose X (standardization occurs along the rows)
    [X, PS] = mapstd(X(1:end, :)); % Get X standardized and obtain STD map
    X = X'; % Transpose X back to orginal form
    X(:, 1) = X(:, 1).^2; % Square height
    X(:, 2) = X(:, 2).^2; % Square width
end

% Iterate for maxIter
for iter = 1:maxIter
    % Pick a random index from all observations
    randIndx = randi([1 n]);
    % Diminshing stepsize (lambda cancels out in main update, but not on
    % regularizer
    stepSize = 1/(lambda * iter);
    % Isolating X and Y for a random ith observation
    y = Y(randIndx, 1);
    x = X(randIndx, 1:end);
    % Dot product of observation x and weight
    prediction = dot(x, w');
    % If prediction does not agree with label
    if y * prediction < 1
        w = ((1 - stepSize*lambda)* w) + (stepSize * y)* x;
    else
        w = ((1 - stepSize*lambda)* w);
    end
end

% Loop through all 14 unlabelled/labelled test images
for index = 1:maxIndex
    % Obtaining filenames
    filenameTestLabelled = strcat(directoryPath, 'testLabelled\',...
        srcTestLabelled(index).name);
    filenameTestUnlabelled = strcat(directoryPath, 'testUnlabelled\',...
        srcTestUnlabelled(index).name);
    
    % Reading filenames as images
    testLabelled = imread(filenameTestLabelled);
    testUnlabelled = imread(filenameTestUnlabelled);
    
    % Converting labelled test images to grayscale (grayscale 84 = hole)
    testLabelledGray = rgb2gray(testLabelled);
    
    % Obtaining image size
    [n, m] = size(testLabelledGray);
    
    % Loop through every pixel in the image
    for height = 1:n
        for width = 1:m
            % If pixel is not hole, paint black
            if testLabelledGray(height,width) ~= 84
                testLabelledGray(height,width) = 0;
            % Else if pixel is hole, paint white
            else
                testLabelledGray(height,width) = 255;
            end
        end
    end
    
    % Removing noise from grayscale test image
    testBWNoNoise = bwareaopen(testLabelledGray, 250);
    
    % Initializing variables
    count = 0; % Count of pixels, X & Y -> single position on the table
    data = zeros(n*m, 6); % Stores X, Y, R, G, B, label of image as a table
    
    % Iterate totalPixelCount by number of pixels in this test image
    totalPixelCount = n*m;
    
     testUnlabelled = uint8(testUnlabelled);
     testBWNoNoise = int8(testBWNoNoise);
    % Loop through every pixel in the image / image -> table
    for height = 1:n
        for width = 1:m
            % Iterate count
            count = count + 1;
            % If labelled test image is marked as hole, label = 1
            if testBWNoNoise(height, width) == 1
                % Storing X, Y, R, G, B data
                data(count, 1) = height;
                data(count, 2) = width;
                data(count, 3) = testUnlabelled(height, width, 1);
                data(count, 4) = testUnlabelled(height, width, 2);
                data(count, 5) = testUnlabelled(height, width, 3);
                data(count, 6) = 1;
            % If labelled test image is marked as not hole, label = -1
            else
                % Storing X, Y, R, G, B data
                data(count, 1) = height;
                data(count, 2) = width;
                data(count, 3) = testUnlabelled(height, width, 1);
                data(count, 4) = testUnlabelled(height, width, 2);
                data(count, 5) = testUnlabelled(height, width, 3);
                data(count, 6) = -1;
            end
        end
    end
    
    % Intializing label column
    predictedLabel = zeros(n*m, 1);
    
    % Check if standardization trick is true
    % If yes, map training standardization and square height/width
    % (middling values near the hole will be close to 0)
    if trickSTD == true
        data = data'; % Transpose data
        data2 = mapstd('apply', data(1:end-1, :), PS); % Get data 
                                %standardized with training data STD map
        data2 = data2'; % Transpose data2 back to orginal form
        data = data'; % Transpose data back to orginal form
        data2(:, 1) = data2(:, 1).^2; % Square height
        data2(:, 2) = data2(:, 2).^2; % Square width
        data = [data2, data(:, end)];
    end
    
    % Initializing variables
    count = 0; % Count of pixels, Count -> X & Y on image
    
    % Loop through label/data to fill in confusion matrix/reference image
    % and to predict labels
    for height = 1:n
        for width = 1:m
            % Iterate count through total number of pixels (n*m)
            count = count + 1;
            % Make prediction (weight * row data)
            predictedLabel(count, 1) = sign(data(count, 1) * w(1, 1) + ...
                data(count, 2) * w(1, 2) + ...
                data(count, 3) * w(1, 3) + ...
                data(count, 4) * w(1, 4) +...
                data(count, 5) * w(1, 5));
            if predictedLabel(count, 1) == -1
                testUnlabelled(height, width, 1) = 255; % R = white
                testUnlabelled(height, width, 2) = 255; % G = white
                testUnlabelled(height, width, 3) = 255;
            end
        end
    end
    
    % Check image saving directory
    if strcmpi(evenOrRandom, 'even') == true
        % Saving visual reference of predictions for dataRandom
        if trickSTD == true
            % Saving visual reference of standardized predictions
            filenameMatrix = sprintf(srcTestUnlabelled(index).name);
            filenamePNGified = strrep(filenameMatrix,'jpg','png');
            nameAndPath = strcat(directoryPath,...
                'testEvenSVMSTD\', filenamePNGified); 
            imwrite(testUnlabelled, nameAndPath);
        else
            % Saving visual reference of standardized predictions
            filenameMatrix = sprintf(srcTestUnlabelled(index).name);
            filenamePNGified = strrep(filenameMatrix,'jpg','png');
            nameAndPath = strcat(directoryPath,...
                'testEvenSVM\', filenamePNGified); 
            imwrite(testUnlabelled, nameAndPath);
        end
    else
        % Saving visual reference of predictions for dataEven
        filenameMatrix = sprintf(srcTestUnlabelled(index).name);
        filenamePNGified = strrep(filenameMatrix,'jpg','png');
        nameAndPath = strcat(directoryPath,...
            'testRandomSVM\', filenamePNGified); 
        imwrite(testUnlabelled, nameAndPath);
    end
    
    % Creating confusion matrix for Avg, Best, Worst
    if index == 1
      confusionMatrixAvg =...
          confusionmat(data(:, 6), predictedLabel) / totalPixelCount;
      confusionMatrixBest =...
          confusionmat(data(:, 6), predictedLabel) / totalPixelCount;
      confusionMatrixWorst =...
          confusionmat(data(:, 6), predictedLabel) / totalPixelCount;
      % Initializing worst name
      worstImageName = sprintf(srcTestUnlabelled(index).name);
      % Initializing best name
      bestImageName = sprintf(srcTestUnlabelled(index).name);
      
    % Updating Avg image performance, best performance, and worst 
    % performance
    else
        % Updating Avg
        confusionMatrixAvg =...
            confusionMatrixAvg +...
            confusionmat(data(:, 6), predictedLabel) / totalPixelCount;
        
        % Temp clone of confusion matrix
        temp = confusionmat(data(:, 6), predictedLabel);
        
        % If new true positive and true negative > old, update best
        if (temp(1, 1) + temp(2, 2)) / totalPixelCount >...
                (confusionMatrixBest(1, 1) +...
                confusionMatrixBest(2, 2)) / totalPixelCount 
            
            % Update new Best
            confusionMatrixBest =...
                confusionmat(data(:, 6), predictedLabel) / totalPixelCount;
            % Update new Best image name
            bestImageName = sprintf(srcTestUnlabelled(index).name);
            
        % If new true positive and true negative < old, update worst
        elseif (temp(1, 1) + temp(2, 2)) / totalPixelCount <...
                (confusionMatrixWorst(1, 1) +...
                confusionMatrixWorst(2, 2)) / totalPixelCount 
            
            % Update new Worst
            confusionMatrixWorst =...
                confusionmat(data(:, 6), predictedLabel) / totalPixelCount;
            % Update new Worst image name
            worstImageName = sprintf(srcTestUnlabelled(index).name);
        end
    end
end
% Converting confusion matrices into percentages
confusionMatrixAvg = (confusionMatrixAvg / maxIndex) * 100;
confusionMatrixBest = confusionMatrixBest * 100;
confusionMatrixWorst = confusionMatrixWorst * 100;
end