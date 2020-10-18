%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Quality of Life Improvement Settings %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
close all
rng(1) % For reproducability

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Change Directories Here First %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

directoryPath = 'C:\Users\Sean\Documents\MATLAB\Project2\';

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Image Preprocessing %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Image Preprocessing');
imagePreProcessing(directoryPath);


%%%%%%%%%%%%%%%%%%%%%%%
%%% Table Generator %%%
%%%%%%%%%%%%%%%%%%%%%%%

disp('Table Generator');
dataEven = tableGeneratorEvenSampling(directoryPath);
dataRandom = tableGeneratorRandomSampling(directoryPath);

%%%%%%%%%%%%%%%%%%%%%%%
%%% Data Tabulation %%%
%%%%%%%%%%%%%%%%%%%%%%%

% Tabulating
disp('Data Tabulation (Even)');
X_even = dataEven(:, 1:end-1);
Y_even = dataEven(:, end);
tabulate(Y_even);

disp('Data Tabulation (Random)');
X_random = dataRandom(:, 1:end-1);
Y_random = dataRandom(:, end);
tabulate(Y_random);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Model Comparisons (Naive Bayes VS SVM w/ dataEven) %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Model Comparisons (Naive Bayes VS SVM, w/ dataEven)');

% Naive Bayes w/ Even Data; Setting Up Parameters
disp('Naive Bayes w/ Even Data');
crossVal = false;
evenOrRandom = 'even';
[confusionMatrixAvg, bestImageName, confusionMatrixBest,...
    worstImageName, confusionMatrixWorst] =...
    runNaiveBayes(X_even, Y_even, directoryPath, crossVal, evenOrRandom)

% SVM w/ Even Data; Setting Up Parameters
disp('SVM w/ Even Data');
lambda = 0.001;
maxIter = 1000000;
trickSTD = false;
evenOrRandom = 'even';
% Training and Testing Model
[confusionMatrixAvg, bestImageName, confusionMatrixBest,...
    worstImageName, confusionMatrixWorst] =...
    runSoftSVM(X_even, Y_even, directoryPath,...
    lambda, maxIter, trickSTD, evenOrRandom)

% Naive Bayes w/ Random Data; Setting Up Parameters
disp('Naive Bayes w/ Random Data');
crossVal = false;
evenOrRandom = 'random';
% Training and Testing Model
[confusionMatrixAvg, bestImageName, confusionMatrixBest,...
    worstImageName, confusionMatrixWorst] =...
    runNaiveBayes(X_even, Y_even, directoryPath, crossVal, evenOrRandom)

% SVM w/ Random Data; Setting Up Parameters
disp('SVM w/ Random Data');
lambda = 0.001;
maxIter = 1000000;
trickSTD = false;
evenOrRandom = 'random';
% Training and Testing Model
[confusionMatrixAvg, bestImageName, confusionMatrixBest,...
    worstImageName, confusionMatrixWorst] =...
    runSoftSVM(X_random, Y_random, directoryPath,...
    lambda, maxIter, trickSTD, evenOrRandom)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Model Comparisons (SVM w/ SVM STD  VS Naive Bayes w/ CrossVal) %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Naive Bayes, w/ Random Data & Cross-Validatition; Setting Up Parameters
disp('Naive Bayes, w/ Random Data & Cross-Validatition');
crossVal = true;
evenOrRandom = 'random';
% Training and Testing Model
[confusionMatrixAvg, bestImageName, confusionMatrixBest,...
    worstImageName, confusionMatrixWorst] =...
    runNaiveBayes(X_random, Y_random, directoryPath, crossVal, evenOrRandom)

% SVM w/ Even Data; Setting Up Parameters
disp('SVM w/ Even Data & STD Trick');
lambda = 0.001;
maxIter = 1000000;
trickSTD = true;
evenOrRandom = 'even';
% Training and Testing Model
[confusionMatrixAvg, bestImageName, confusionMatrixBest,...
    worstImageName, confusionMatrixWorst] =...
    runSoftSVM(X_even, Y_even, directoryPath,...
    lambda, maxIter, trickSTD, evenOrRandom)