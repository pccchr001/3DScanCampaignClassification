#include "SVM.h"

SVM::SVM (){
}

SVM::SVM (int nClasses){
  nclasses = nClasses;
}

SVM::~SVM (){
}

void SVM::loadTrainingData(std::vector<std::vector<float>> & dataVec){


  for (int i = 0; i < dataVec.size(); ++i)
    {
      cv::Mat tmp_data(1, dataVec[i].size()-1, CV_32FC1);

      for (u_int j = 0; j < dataVec[i].size()-1; ++j){
          tmp_data.at<float>(0, j) = dataVec[i].at(j);
        }

      training_features.push_back(tmp_data);
      training_labels.push_back(dataVec[i].at(dataVec[i].size()-1));
    }
}


void SVM::train(bool tuning, bool featureselection)
{
  svm = cv::ml::SVM::create();

  training_data = cv::ml::TrainData::create(training_features, cv::ml::ROW_SAMPLE, cv::Mat(training_labels),
                                            cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray());

  svm->setType(cv::ml::SVM::C_SVC);
  svm->setKernel(cv::ml::SVM::RBF);

  std::cout << "Training SVM with all features (" << training_data->getNTrainSamples() << " samples, "
            << training_data->getNVars() << " features)..." << std::endl;

  double t = (double) cv::getTickCount();

  cv::ml::ParamGrid ignoreGrid;
  ignoreGrid.logStep =0;


  auto startTrain = std::chrono::high_resolution_clock::now();
  svm->trainAuto(training_data, 5); // always use trainauto
  auto finishTrain = std::chrono::high_resolution_clock::now();
  trainTime = finishTrain - startTrain;

  std::cout << "Finished training SVM. Gamma: " << svm->getGamma() << ". C: " << svm->getC() << std::endl;

  t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
  printf("Training took %i s\n",(int)t);
  printf( "Train error: %f\n", svm->calcError(training_data, false, cv::noArray()));
  printf( "Test error: %f\n\n", svm->calcError(training_data, true, cv::noArray()));

  saveModel("../data/models/classifier.yml");
}


void SVM::saveModel(std::string filename){
  svm->save(filename);
}


void SVM::loadModel(std::string filename){
  svm = cv::ml::StatModel::load<cv::ml::SVM>(filename);
}

void SVM::setMeanStd(std::vector<float> m, std::vector<float> s){
  means = m;
  stds = s;
}

std::vector<int> SVM::predict(std::vector<std::vector<float>> testVectors, bool tuning){
  std::vector<int> resultsVector;

  // Normalize test vector
  for (int i = 0; i < testVectors.size() ; ++i){
      for (int j = 0; j < testVectors[i].size(); ++j)
        testVectors[i][j] = (testVectors[i][j]-means[j])/stds[j];}

auto startTest = std::chrono::high_resolution_clock::now();
  for (u_int i = 0 ; i < testVectors.size(); ++i){
      testVectors[i].pop_back(); // remove label from vector
      int result = svm->predict(testVectors[i]);
      resultsVector.push_back(result);
    }
  auto finishTest = std::chrono::high_resolution_clock::now();
  predictTime += finishTest - startTest;

  return resultsVector;
}

std::chrono::duration<double> SVM::getTraintime(){
  return trainTime;

}
std::chrono::duration<double> SVM::getPredicttime(){
  return predictTime;
}



