#ifndef CloudFex_H_
#define CloudFex_H_

#include "Common.h"
#include <fstream>
#include <boost/assign/std/vector.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include "pcl/segmentation/supervoxel_clustering.h"

class CloudFex {
public:
  CloudFex();
  virtual ~CloudFex();
  void calcPointFeat(pcl::PointXYZRGBA &,std::vector<int> &,std::vector<float> &,int &,int &, std::vector<float> *);
  void calcVoxelFeat(pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr &, std::vector<int>);
  void setPointMatrix (Eigen::MatrixXd [], int);
  void setCentroidCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &);
  void saveFeatures(std::string);
  void clearFeatures();
  void saveExamples(std::string);
  std::vector<float> getFeatures();

private:
  std::vector<float> features;
  Eigen::MatrixXd * points;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr centroid_cloud;
  int max_level;
};

#endif
