#ifndef SVM_H_
#define SVM_H_

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

#include <iomanip>
#include <fstream>

class SVM
{
  public:
    SVM ();
    SVM (int);
    void train(bool,bool);
    void saveModel(std::string);
    void loadModel(std::string);
    void setMeanStd(std::vector<float>, std::vector<float>);
    void loadTrainingData(std::vector<std::vector<float>> &);
    std::vector<int> predict(std::vector<std::vector<float>>, bool);
    std::chrono::duration<double> getTraintime();
    std::chrono::duration<double> getPredicttime();
    virtual ~SVM ();

  private:
    int nclasses;
    cv::Mat training_features;
    std::vector<int> training_labels;
    cv::Ptr<cv::ml::TrainData> training_data;
    cv::Ptr<cv::ml::SVM> svm;
    std::vector<float> means;
    std::vector<float> stds;
    std::chrono::duration<double> predictTime;
    std::chrono::duration<double> trainTime;

};

#endif
