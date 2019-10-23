#include "NeuralNet.h"

NeuralNet::NeuralNet (){
}

NeuralNet::NeuralNet (int nClasses){
  nclasses = nClasses;
}

NeuralNet::~NeuralNet (){
}

void NeuralNet::loadTrainingData(std::vector<std::vector<float>> & dataVec){

  for (int i = 0; i < dataVec.size(); ++i){
      cv::Mat tmp_data(1, dataVec[i].size()-1, CV_32FC1);

      for (u_int j = 0; j < dataVec[i].size()-1; ++j){
          tmp_data.at<float>(0, j) = dataVec[i].at(j);
        }

      training_features.push_back(tmp_data);
      cv::Mat_<float> trainClasses = cv::Mat::zeros( 1, nclasses, CV_32FC1 );
      trainClasses.at<float>(0, dataVec[i].at(dataVec[i].size()-1)) = 1.f;
      training_labels.push_back(trainClasses);
    }
}

void NeuralNet::setMeanStd(std::vector<float> m, std::vector<float> s){
  means = m;
  stds = s;
}


void NeuralNet::train(bool tuning, bool featureselection)
{
  // Shuffle samples and labels to remove bias
  std::vector <int> seeds;
  for (int i = 0; i < training_features.rows; i++)
    seeds.push_back(i);

  cv::randShuffle(seeds);

  cv::Mat shuffled_feat;
  cv::Mat shuffled_labels;
  for (int i = 0; i < training_features.rows; i++)
    {
      shuffled_feat.push_back(training_features.row(seeds[i]));
      shuffled_labels.push_back(training_labels.row(seeds[i]));
    }

  training_features = shuffled_feat;
  training_labels = shuffled_labels;

  // Create the neural network
  cv::Mat_<int> layerSizes(3, 1);
  layerSizes(0, 0) = training_features.cols;
  layerSizes(0, 1) = (training_features.cols+training_labels.cols)/2;
  layerSizes(0, 2) = training_labels.cols;

  neuralnet = cv::ml::ANN_MLP::create();
  neuralnet->setLayerSizes(layerSizes);

  neuralnet->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 0, 0); // output range: [-1.7159, 1.7159]
  neuralnet->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 10000, 0.0001));
  neuralnet->setTrainMethod(cv::ml::ANN_MLP::RPROP, 0.01);

  training_data = cv::ml::TrainData::create(training_features, cv::ml::ROW_SAMPLE, training_labels);

  std::cout << "Training NeuralNet with all features (" << training_data->getNTrainSamples() << " samples, "
            << training_data->getNVars() << " features)..." << std::endl;

  auto startTrain = std::chrono::high_resolution_clock::now();
  neuralnet->train(training_data);
  auto finishTrain = std::chrono::high_resolution_clock::now();
  trainTime = finishTrain - startTrain;

  if (neuralnet->isTrained()){
      saveModel("../data/models/classifier.yml");
    }

  std::cout << "Finished training NeuralNet." << std::endl;
}


void NeuralNet::saveModel(std::string filename){
  neuralnet->save(filename);
}

void NeuralNet::loadModel(std::string filename){
  neuralnet = cv::ml::StatModel::load<cv::ml::ANN_MLP>(filename);
}

std::vector<int> NeuralNet::predict(std::vector<std::vector<float>> testVectors, bool tuning){
  std::vector<int> resultsVector;

  for (int i = 0; i < testVectors.size() ; ++i){
      for (int j = 0; j < testVectors[i].size(); ++j)
        testVectors[i][j] = (testVectors[i][j]-means[j])/stds[j];
    }

  auto startTest = std::chrono::high_resolution_clock::now();
  for (u_int i = 0 ; i < testVectors.size(); ++i){
      testVectors[i].pop_back(); // remove label from vector
      int result = neuralnet->predict(testVectors[i]);
      resultsVector.push_back(result);
    }
  auto finishTest = std::chrono::high_resolution_clock::now();
  predictTime += finishTest - startTest;

  return resultsVector;
}

std::chrono::duration<double> NeuralNet::getTraintime(){
  return trainTime;

}
std::chrono::duration<double> NeuralNet::getPredicttime(){
  return predictTime;
}
