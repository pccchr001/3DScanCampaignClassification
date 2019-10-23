#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

class NeuralNet
{
  public:
    NeuralNet ();
    NeuralNet (int);
    void train(bool,bool);
    void saveModel(std::string);
    void loadModel(std::string);
    void loadTrainingData(std::vector<std::vector<float>> &);
    void setMeanStd(std::vector<float>, std::vector<float>);
    std::vector<int> predict(std::vector<std::vector<float>>,bool);
    std::chrono::duration<double> getTraintime();
    std::chrono::duration<double> getPredicttime();
    virtual ~NeuralNet ();

  private:
    int nclasses;
    cv::Mat training_features;
    cv::Mat training_labels;
    cv::Ptr<cv::ml::TrainData> training_data;
    cv::Ptr<cv::ml::ANN_MLP> neuralnet;
    std::vector<float> means;
    std::vector<float> stds;
    std::chrono::duration<double> predictTime;
    std::chrono::duration<double> trainTime;
};

#endif
