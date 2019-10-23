#ifndef TRAININGLABELS_H_
#define TRAININGLABELS

#include "Common.h"

class TrainingLabels {

private:
  std::vector<std::vector<int>> v;
  int ns;
  int nc;

public:
  TrainingLabels(int,int);
  virtual ~TrainingLabels();
  void addLabel (int,int, float);
  void printTotals();
  void overSample(int);
  void underSample(int);
  void combine(TrainingLabels);
  int getClassTotal(int);
  std::vector<int> getExamples(int, int);
};

#endif
