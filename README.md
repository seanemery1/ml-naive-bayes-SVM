## INSTRUCTIONS
1) open runScripts.m

2) modify line 13 of runScripts.m - change 'directoryPath' to the project folder

3) run runScripts.m

## FUNCTIONS (Inputs/Outputs)
1) imagePreProcessingRandomSampling.m
in: images
out: pixel location-label matrices for both randomly and evenly sampled data from each image

2) tableGeneratorRandomSampling.m
in: training images and location-label matrices
out: resultsRandom.mat (training set matrix with pixel rows & feature/label columns)

3) tableGeneratorEvenSampling.m
in: training images and location-label matrices
out: resultsRandom.mat (training set matrix with pixel rows & feature/label columns)

4) runNaiveBayes.m
in: training set matrix, test images, cross-validation bool
out: prediction images, performance stats (confusion matrices for accuracy/misclassification rates, best performing & worst performing image)

5) runSoftSVM.m
in: training set matrix, test images, cross-validation bool, data standardization bool
out: prediction images, performance stats (confusion matrices for accuracy/misclassification rates, best performing & worst performing image)
