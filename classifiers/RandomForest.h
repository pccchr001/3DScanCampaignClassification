#ifndef RANDOMFOREST_H_
#define RANDOMFOREST_H_

#include "ArgumentHandler.h"
#include "ForestClassification.h"
#include "utility.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

class RandomForest
{
  public:
    RandomForest();
    RandomForest (int);
    virtual ~RandomForest ();
    void saveModel(std::string);
    void loadModel(std::string);
    void loadTrainingData(std::vector<std::vector<float>> &);
    void setMeanStd(std::vector<float>, std::vector<float>);
    void train(bool,bool);
    void trimTrainingVectors(std::vector<int>&);
    std::vector<int> predict(std::vector<std::vector<float>>&, bool);
    std::chrono::duration<double> getTraintime();
    std::chrono::duration<double> getPredicttime();

  private:
    int nclasses;
    std::vector<std::vector<float>> trainingVectors;
    int nSelect;
    std::vector<float> means;
    std::vector<float> stds;
    std::chrono::duration<double> predictTime;
    std::chrono::duration<double> trainTime;

};

#endif
