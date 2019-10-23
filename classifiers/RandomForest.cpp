#include "RandomForest.h"

RandomForest::RandomForest (){
}

RandomForest::RandomForest (int nClasses){
  nclasses = nClasses;
}

RandomForest::~RandomForest (){
}

void RandomForest::loadTrainingData(std::vector<std::vector<float>> & dataVec){
  trainingVectors = dataVec;

}

void RandomForest::setMeanStd(std::vector<float> m, std::vector<float> s){
  means = m;
  stds = s;
}

void RandomForest::train(bool tuneParams, bool selectImp)
{
  int c; char **v;
  ForestClassification forest;
  std::vector<float> bestParams;

  ArgumentHandler arg_handler(c,v);
  arg_handler.file = "../data/samples/training.data";
  arg_handler.depvarname = "Class";

  arg_handler.ntree = 100;
  arg_handler.targetpartitionsize = 25;

  arg_handler.nthreads = std::thread::hardware_concurrency();
  arg_handler.verbose = true;
  arg_handler.write = true;
  arg_handler.savemem = false;
  arg_handler.outprefix = "../data/models/ranger_out";

  try {

    std::ostream* verbose_out;

    // Verbose output to logfile if non-verbose mode
    if (arg_handler.verbose) {
        verbose_out = &std::cout;
      } else {
        std::ofstream* logfile = new std::ofstream();
        logfile->open(arg_handler.outprefix + ".log");
        if (!logfile->good()) {
            throw std::runtime_error("Could not write to logfile.");
          }
        verbose_out = logfile;
      }


    if (tuneParams){

        std::vector<float> nTrees {25,50,100};
        std::vector<float> nodeSize {1,10,25};
        arg_handler.outprefix = "../data/models/tuning";

        // Shuffle and split samples into training and testing sets
        float ratio = 0.8;
        std::random_shuffle (trainingVectors.begin(), trainingVectors.end()); // Shuffle samples
        std::size_t const train_size = trainingVectors.size() * ratio;
        std::vector<std::vector<float>> trainSet(trainingVectors.begin(), trainingVectors.begin() + train_size);
        std::vector<std::vector<float>> testSet(trainingVectors.begin() + train_size, trainingVectors.end());

        // Determine best-scoring parameter combination
        float bestScore = 0;

        for (u_int i = 0; i < nTrees.size(); ++i)
          for (u_int j = 0; j < nodeSize.size(); ++j)
            {
              ForestClassification tuningForest;
              arg_handler.ntree = nTrees[i];
              arg_handler.targetpartitionsize = nodeSize[j];

              arg_handler.checkArguments();
              tuningForest.initCppFromVec(arg_handler.depvarname, arg_handler.memmode, trainSet, arg_handler.mtry,
                                           arg_handler.outprefix, arg_handler.ntree, verbose_out, arg_handler.seed, arg_handler.nthreads,
                                           arg_handler.predict, arg_handler.impmeasure, arg_handler.targetpartitionsize, arg_handler.splitweights,
                                           arg_handler.alwayssplitvars, arg_handler.statusvarname, arg_handler.replace, arg_handler.catvars,
                                           arg_handler.savemem, arg_handler.splitrule, arg_handler.caseweights, arg_handler.predall, arg_handler.fraction,
                                           arg_handler.alpha, arg_handler.minprop, arg_handler.holdout, arg_handler.predictiontype,
                                           arg_handler.randomsplits);

              std::cout << "Training with nTrees " << nTrees[i] << ", nodeSize " << nodeSize[j] << "... " << std::endl;


              tuningForest.run(true);


              tuningForest.saveToFile();
              tuningForest.writeOutput();

              // Predict test set and update bestParams if score improves
              std::cout << "Predicting test set..." << std::endl;

              std::vector<int> testResults = predict(testSet, true);
              float correct = 0;
              for (size_t i = 0; i < testResults.size(); ++i)
                if (testResults[i] == testSet[i].back())
                  correct++;

              float score = correct/testResults.size()*100;

              if (correct/testResults.size()*100 > bestScore){
                  bestScore = score;
                  bestParams = {nTrees[i],nodeSize[j]};
                  std::cout << "Correctly classified " << score << "% of test set (New best!)." << std::endl;
                  tuningForest.saveToFile("../data/models/ranger_out.forest"); // Only saves to this file if score has improved
                }
              else
                std::cout << "Correctly classified " << score << "% of test set. (Needed to beat " << bestScore << ")." << std::endl;

            }

        std::cout << "Finished tuning parameters, best combination was: nTrees "
                  << bestParams[0] << ", nodeSize " << bestParams[1] << std::endl;

        // Set best parameters for remaining code
        arg_handler.ntree = bestParams[0];
        arg_handler.targetpartitionsize = bestParams[1];
        arg_handler.outprefix = "../data/models/ranger_out"; // Set this back to original now that we're done tuning

      }

    if (selectImp) {

        int nSelect = 25;
        arg_handler.impmeasure = IMP_GINI; // This tells ranger to calculate variable importance (gini)
        ForestClassification impForest;
        arg_handler.checkArguments();

        impForest.initCppFromVec(arg_handler.depvarname, arg_handler.memmode, trainingVectors, arg_handler.mtry,
                               arg_handler.outprefix, arg_handler.ntree, verbose_out, arg_handler.seed, arg_handler.nthreads,
                               arg_handler.predict, arg_handler.impmeasure, arg_handler.targetpartitionsize, arg_handler.splitweights,
                               arg_handler.alwayssplitvars, arg_handler.statusvarname, arg_handler.replace, arg_handler.catvars,
                               arg_handler.savemem, arg_handler.splitrule, arg_handler.caseweights, arg_handler.predall, arg_handler.fraction,
                               arg_handler.alpha, arg_handler.minprop, arg_handler.holdout, arg_handler.predictiontype,
                               arg_handler.randomsplits);

        std::cout << "Training with nTrees " << arg_handler.ntree << ", nodeSize "
                  << arg_handler.targetpartitionsize << ", impMeasure " << arg_handler.impmeasure << std::endl;
        impForest.run(true);
        std::vector<double> importances = impForest.getVariableImportance();

        std::vector<int> selectedFeat(importances.size());
        std::vector<double> sorted = importances;

        std::sort(sorted.begin(), sorted.end());

        for (int s = 0; s < importances.size(); ++s){
            if (importances.at(s) >= sorted.at(sorted.size()-nSelect))
              {
                std::cout << "Chose feature #" << s << " with importance: " << importances.at(s) << std::endl;
                selectedFeat.at(s) = 1;
              }
            else
              {
                std::cout << "DIDN'T choose #" << s << " with importance: " << importances.at(s) << std::endl;
              }
          }

        std::ofstream ofile;
        ofile.open("/media/chris/Seagate Backup Plus Drive/results/importantFeat.txt");
        for (int i = 0; i < selectedFeat.size(); ++i)
          ofile << selectedFeat[i] << "\n";
        ofile.close();

        ofile.open("/media/chris/Seagate Backup Plus Drive/results/importantVals.txt");
        for (int i = 0; i < selectedFeat.size(); ++i)
          ofile << importances[i] << "\n";
        ofile.close();

        trimTrainingVectors(selectedFeat); // Trim the training vectors to only have selected features in them

        arg_handler.impmeasure = IMP_NONE; // Turn this off now that we have already selected best features

        ForestClassification impForest2;
        arg_handler.checkArguments();
        impForest2.initCppFromVec(arg_handler.depvarname, arg_handler.memmode, trainingVectors, arg_handler.mtry,
                               arg_handler.outprefix, arg_handler.ntree, verbose_out, arg_handler.seed, arg_handler.nthreads,
                               arg_handler.predict, arg_handler.impmeasure, arg_handler.targetpartitionsize, arg_handler.splitweights,
                               arg_handler.alwayssplitvars, arg_handler.statusvarname, arg_handler.replace, arg_handler.catvars,
                               arg_handler.savemem, arg_handler.splitrule, arg_handler.caseweights, arg_handler.predall, arg_handler.fraction,
                               arg_handler.alpha, arg_handler.minprop, arg_handler.holdout, arg_handler.predictiontype,
                               arg_handler.randomsplits);

        std::cout << "\nTraining with " << nSelect << " best features. Parameters: nTrees " << arg_handler.ntree << ", nodeSize " << arg_handler.targetpartitionsize << std::endl;


        auto startTrain = std::chrono::high_resolution_clock::now();
        impForest2.run(true);
        auto finishTrain = std::chrono::high_resolution_clock::now();
        trainTime = finishTrain - startTrain;

        impForest2.saveToFile();
        impForest2.writeOutput();
      }

    if (!tuneParams && !selectImp){ // if neither tuning params nor selecting features

        arg_handler.checkArguments();
        forest.initCppFromVec(arg_handler.depvarname, arg_handler.memmode, trainingVectors, arg_handler.mtry,
                               arg_handler.outprefix, arg_handler.ntree, verbose_out, arg_handler.seed, arg_handler.nthreads,
                               arg_handler.predict, arg_handler.impmeasure, arg_handler.targetpartitionsize, arg_handler.splitweights,
                               arg_handler.alwayssplitvars, arg_handler.statusvarname, arg_handler.replace, arg_handler.catvars,
                               arg_handler.savemem, arg_handler.splitrule, arg_handler.caseweights, arg_handler.predall, arg_handler.fraction,
                               arg_handler.alpha, arg_handler.minprop, arg_handler.holdout, arg_handler.predictiontype,
                               arg_handler.randomsplits);

        auto startTrain = std::chrono::high_resolution_clock::now();
        forest.run(true);
        auto finishTrain = std::chrono::high_resolution_clock::now();
        trainTime = finishTrain - startTrain;

        forest.saveToFile();
        forest.writeOutput();
      }

  }
  catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
  }
}


std::vector<int> RandomForest::predict(std::vector<std::vector<float>>& testVectors, bool tuningForest)
{
  // Normalize test vectors
  for (int i = 0; i < testVectors.size() ; ++i){
      for (int j = 0; j < testVectors[i].size(); ++j)
        testVectors[i][j] = (testVectors[i][j]-means[j])/stds[j];
    }

  int c; char **v;
  ArgumentHandler arg_handler(c,v);
  std::vector<int> resultsVector;
  arg_handler.verbose = false;
  arg_handler.savemem = true;
  tuningForest? arg_handler.predict = "../data/models/tuning.forest" : arg_handler.predict = "../data/models/ranger_out.forest";

  ForestClassification forest;

  try{
    arg_handler.checkArguments();
    std::ostream* verbose_out;

    if (arg_handler.verbose) {
        verbose_out = &std::cout;
      } else{
        std::ofstream* logfile = new std::ofstream();
        logfile->open(arg_handler.outprefix + ".log");
        verbose_out = logfile;
      }

    forest.initCppFromVec(arg_handler.depvarname, arg_handler.memmode, testVectors, arg_handler.mtry,
                           arg_handler.outprefix, arg_handler.ntree, verbose_out, arg_handler.seed, arg_handler.nthreads,
                           arg_handler.predict, arg_handler.impmeasure, arg_handler.targetpartitionsize, arg_handler.splitweights,
                           arg_handler.alwayssplitvars, arg_handler.statusvarname, arg_handler.replace, arg_handler.catvars,
                           arg_handler.savemem, arg_handler.splitrule, arg_handler.caseweights, arg_handler.predall, arg_handler.fraction,
                           arg_handler.alpha, arg_handler.minprop, arg_handler.holdout, arg_handler.predictiontype,
                           arg_handler.randomsplits);


    auto startTest = std::chrono::high_resolution_clock::now();
    forest.run(false);
    auto finishTest = std::chrono::high_resolution_clock::now();
    predictTime += finishTest - startTest;

    // Get prediction results
    std::vector<std::vector<std::vector<double>>> predictions = forest.getPredictions();
    int cnt = 0;
    for (size_t i = 0; i < predictions.size(); ++i)
      for (size_t j = 0; j < predictions[i].size(); ++j)
        for (size_t k = 0; k < predictions[i][j].size(); ++k) {
            resultsVector.push_back(int(predictions[i][j][k]));
            cnt++;
          }
  }

  catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
  }

  return resultsVector;
}

void RandomForest::trimTrainingVectors(std::vector<int> &selected){

  std::vector<std::vector<float>> tempVectors;

  for (int v = 0; v < trainingVectors.size(); ++v)
    {
      std::vector<float> tempVector;
      for (int s = 0; s < selected.size(); ++s){
          if (selected[s] == 1)
            tempVector.push_back(trainingVectors[v][s]);
        }
      tempVector.push_back(trainingVectors[v].back()); // append label
      tempVectors.push_back(tempVector);
    }
  trainingVectors = tempVectors;
}

void RandomForest::loadModel(std::string filename){
  // Note this classifier loads trained model in predict method
}

void RandomForest::saveModel(std::string filename){
  // Note this classifier saves trained model in train method
}

std::chrono::duration<double> RandomForest::getTraintime(){
  return trainTime;

}
std::chrono::duration<double> RandomForest::getPredicttime(){
  return predictTime;
}
