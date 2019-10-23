#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>

class TrainingData
{
public:
  TrainingData ();
  virtual ~TrainingData ();
  void addTrainingData(std::vector<float>&);
  void normalizeFeat();
  void writeTrainingData(std::string,bool);

  std::vector<std::vector<float>> & getTrainingData();
  std::vector<float> getMeans();
  std::vector<float> getStds();

  void setTrainingData(std::vector<std::vector<float>>);
  void setMeans(std::vector<float>);
  void setStds(std::vector<float>);

private:
  std::vector<std::vector<float>> trainingVectors;
  std::vector<float> means;
  std::vector<float> stds;
};


#endif // TRAININGDATA_H
