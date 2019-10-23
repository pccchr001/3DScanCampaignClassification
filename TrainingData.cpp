#include "TrainingData.h"


TrainingData::TrainingData (){
}

TrainingData::~TrainingData (){
}

void TrainingData::addTrainingData(std::vector<float>& td){
  trainingVectors.push_back(td);
}

std::vector<std::vector<float>>& TrainingData::getTrainingData(){
  return trainingVectors;
}

std::vector<float> TrainingData::getMeans(){
  return means;
}

std::vector<float> TrainingData::getStds(){
  return stds;
}


void TrainingData::setTrainingData(std::vector<std::vector<float>> vecs){
  trainingVectors = vecs;
}

void TrainingData::setMeans(std::vector<float> m){
  means = m;
}

void TrainingData::setStds(std::vector<float> s){
  stds = s;
}

void TrainingData::writeTrainingData(std::string filename, bool append){

  std::ofstream outFile;

  if (append){
      outFile.open(filename, std::ios::app);
    }
  else
    outFile.open(filename);

  for (size_t i = 0; i < trainingVectors.size(); ++i){
      for (size_t j = 0; j < trainingVectors[i].size(); ++j){
          outFile << std::setprecision(5) << trainingVectors[i][j] << ",";
        }
      outFile << std::endl;
    }
  outFile.close();

}

// Normalize with mean standardization: x' = (x-mean)/std
void TrainingData::normalizeFeat(){
  cv::Scalar mean;
  cv::Scalar std;

  for (int i = 0; i < trainingVectors[0].size()-1 ; ++i){

      std::vector<float> tempCol(trainingVectors.size());

      for (int j = 0; j < trainingVectors.size() ; ++j){
          tempCol[j] = trainingVectors[j][i];
        }

      cv::meanStdDev(tempCol, mean,std);
      means.push_back(mean[0]);
      stds.push_back(std[0]);

      for (int j = 0; j < trainingVectors.size() ; ++j)
        trainingVectors[j][i] = (trainingVectors[j][i]-mean[0])/std[0];

    }
}
