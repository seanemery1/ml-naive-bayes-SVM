function [confusionMatrixAvg, bestImageName, confusionMatrixBest,...
    worstImageName, confusionMatrixWorst] =...
    runNaiveBayes(X, Y, directoryPath, crossVal, evenOrRandom)
% Runs MatLab's built in Naive-Bayes function. Optional parameter of 
% cross-validation. 

% Initializing directory names
srcTestLabelled =...
    dir(strcat(directoryPath, 'testLabelled\*.jpg'));
srcTestUnlabelled =...
    dir(strcat(directoryPath, 'testUnlabelled\*.jpg'));

% Check if CrossVal is true or false
if crossVal == true
    mdl = fitcnb(X, Y, 'CrossVal', 'on');
else
    mdl = fitcnb(X, Y);
end

% Initializing max index as number of images in testLabelled/testUnlabelled
maxIndex = max(size(srcTestUnlabelled)); 

% Initializing total pixel count
totalPixelCount = 0;

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
    
    
    % Predict labels and crossVal check
    if crossVal == true
        predictedLabel = predict(mdl.Trained{1}, data(:, 1:end-1));
    else
        predictedLabel = predict(mdl, data(:, 1:end-1));
    end
        
    
    % Initializing variables
    count = 0; % Count of pixels, Count -> X & Y on image
    
    % Loop through label/data to fill in confusion matrix/reference image
    for height = 1:n
        for width = 1:m
            % Iterate count
            count = count + 1;
            % If predicted label is hole, recolor pixel as orignal
            if predictedLabel(count, 1) == 1
                testUnlabelled(height, width, 1) = data(count, 3); % R
                testUnlabelled(height, width, 2) = data(count, 4); % G
                testUnlabelled(height, width, 3) = data(count, 5) ; % B

            % If predicted label is not hole, recolor pixel as white
            elseif predictedLabel(count, 1) == -1
                testUnlabelled(height, width, 1) = 255; % R = white
                testUnlabelled(height, width, 2) = 255; % G = white
                testUnlabelled(height, width, 3) = 255; % B = white
            end
        end
    end
    
    % Check image saving directory
    if strcmpi(evenOrRandom, 'random') == true
        % Saving visual reference of predictions for dataEven
        if crossVal == true
            % Saving visual reference of cross-validated predictions
            filenameMatrix = sprintf(srcTestUnlabelled(index).name);
            filenamePNGified = strrep(filenameMatrix,'jpg','png');
            nameAndPath = strcat(directoryPath,...
                'testRandomNBCrossVal\', filenamePNGified); 
            imwrite(testUnlabelled, nameAndPath);
        else
            % Saving visual reference of non cross-validated predictions

            filenameMatrix = sprintf(srcTestUnlabelled(index).name);
            filenamePNGified = strrep(filenameMatrix,'jpg','png');
            nameAndPath = strcat(directoryPath,...
                'testRandomNB\', filenamePNGified); 
            imwrite(testUnlabelled, nameAndPath);
        end
    else
        % Saving visual reference of predictions for dataRandom
        filenameMatrix = sprintf(srcTestUnlabelled(index).name);
        filenamePNGified = strrep(filenameMatrix,'jpg','png');
        nameAndPath = strcat(directoryPath,...
            'testEvenNB\', filenamePNGified); 
        imwrite(testUnlabelled, nameAndPath);

    end
    
    % Creating confusion matrix
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
                confusionmat(data(:, 6), predictedLabel) /totalPixelCount;
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