#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// MLPack headers
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/kmeans/allow_empty_clusters.hpp>
#include <armadillo>

class Clustering {

public:
  Clustering();
  void setSamples(cv::Mat & samples);
  void kmeans (int,int);
  void kmeansNoEmpty (int, int);
  void dbscan (double,int);
  void genColors(int);
  void saveClusters(std::string, cv::Mat &, cv::Mat &, int);
  void clearVecs();

  int n_clusters;
  arma::mat trainingData;
  cv::Mat centers;
  std::vector<int> labels;
  std::vector<int> clusterSizes;
  std::vector<std::vector<int>> clusterIndices;
  std::vector<std::vector<int>> colors;

};

#endif
