#include "TrainingLabels.h"

TrainingLabels::TrainingLabels(int n_classes, int n_scans) {
  nc = n_classes;
  ns = n_scans;
  v.resize(nc*ns);
}

TrainingLabels::~TrainingLabels() {
}

void TrainingLabels::addLabel (int label,int scan_id, float idx){
  v.at(label*ns + scan_id).push_back(idx);
}

// Samples examples until max_examples is met
void TrainingLabels::underSample(int max_examples){

  std::vector<std::vector<int>> temp;;
  temp.resize(v.size());
  srand (time(NULL));

  for (int i = 0; i < nc; ++i) // For each class
    {
      int classTotal = getClassTotal(i);

      if (classTotal > 0 && classTotal >= max_examples) // Check if this class has enough examples to sample from
        {
          int cnt = 0;
          while (cnt < max_examples)
            {
              int random_sample = rand() % (classTotal - cnt); // Get random sample from classTotal (minus cnt to account for erased samples)
              int total = 0;

              for (int j = 0; j < ns; ++j) // For each scan
                {
                  total += v.at(i*ns + j).size();
                  if (random_sample < total) // If random sample is less than total, we know the sample is from this scan
                    {
                      int idx = random_sample - (total - v.at(i*ns + j).size()); // Get the random samples index (in this scan)
                      temp.at(i*ns + j).push_back(v.at(i*ns + j).at(idx)); // Add sample to temp training data
                      v.at(i*ns + j).erase (v.at(i*ns + j).begin()+idx); // Delete sample from training data
                      cnt++;
                      break;
                    }
                }
            }
        }
    }
  v = temp;
}

// Add copies of instances from the under-represented classes
void TrainingLabels::overSample(int min_examples)
{
  srand (time(NULL));
  std::vector<std::vector<int>> temp = v; // Create a temp copy of data that we will push duplicates onto

  for (int i = 0; i < nc; ++i) // For each class
    {
      int classTotal = getClassTotal(i);

      if (classTotal > 0 && classTotal < min_examples) // If class total is greater than maximum
        {
          int diff = min_examples - classTotal;
          int cnt = 0;

          while (cnt < diff)
            {
              int random_sample = rand() % classTotal; // Get random sample from classTotal
              int total = 0;
              for (int j = 0; j < ns; ++j) // For each scan
                {
                  total += v.at(i*ns + j).size();
                  if (random_sample < total) // If random sample is less than total, we know the sample is from this scan
                    {
                      int idx = random_sample - (total - v.at(i*ns + j).size()); // Get the random samples index (in this scan)
                      temp.at(i*ns + j).push_back(v.at(i*ns + j).at(idx)); // Add sample to training data
                      cnt++;
                      break;
                    }
                }
            }
        }
    }
  v = temp;
}


void TrainingLabels::combine(TrainingLabels rhs){
  for (int i = 0; i < nc; ++i){ // For each class
      for (int j = 0; j < ns; ++j){ // For each scan
          std::vector<int> examples = rhs.getExamples(i,j); // Get rhs examples
          for (u_int k = 0; k < examples.size(); ++k) // For each example
            v.at(i*ns + j).push_back(examples[k]); // Add example to lhs vector
        }
    }
}

int TrainingLabels::getClassTotal(int c){
  int total = 0;

  for (int i = 0; i < ns; ++i)
    total += getExamples(c,i).size();

  return total;
}

// Prints number of training examples per class
void TrainingLabels::printTotals(){
  int totals[nc];

  for (int i = 0; i < nc; ++i){
      totals[i] = 0;
      for (int j = 0; j < ns; ++j){
          std::cout << "Scan " << j << " Class " << i << " nSamples: " << getExamples(i,j).size() << std::endl;
          totals[i] += getExamples(i,j).size(); // Size of vector of training examples indices
        }
    }

  std::cout << "Class totals: " << std::endl;
  for (int i = 0; i < nc; ++i)
    std::cout << "Class " << i+1  << ": " << totals[i] << std::endl;
  std::cout << std::endl;
}

// Returns vector of training example indices for a given class in a given scan
std::vector<int> TrainingLabels::getExamples(int label, int scan_id){
  return v.at(label*ns + scan_id);
}
