# 3D Scan Campaign Classification with Representative Training Scan Selection

This repository hosts the framework written for the master's thesis titled "3D Scan Campaign Classification with Representative Training Scan Selection". The code is intended as a reference for the implementation sections of the thesis.

A summary of important files is given below:
* __CloudFex.cpp__ - Feature extraction from point neighbourhoods and supervoxel segments.
* classifiers/__RandomForest.cpp__, __SVM.cpp__, __NeuralNet.cpp__ - Implementation of tested RF, SVM and MLP classifiers.
* __ScanSelection.cpp__ - Implementation of designed _balanced_, _similarity_ and _distinct_ selection schemes.
* __Clustering.cpp__ & __Segmenter.cpp__  - Contain k-means clustering and VCCS segmentation methods used by scan selection.
* __Main.cpp__ - Contains main method for running the scan classification pipeline.
* scripts/__fs.py__ - Feature selection from extracted features using scikit-learn filter methods.
* scripts/__analysis.py__ - Generates various accuracy statistics from classification results file.


