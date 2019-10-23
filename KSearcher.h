#ifndef KSearcherER_H_
#define KSearcherER_H_

#include "Common.h"
#include "pcl/kdtree/impl/kdtree_flann.hpp"

class KSearcher {

private:
  pcl::KdTreeFLANN<pcl::PointXYZRGBA, flann::L2_Simple<float>> kdtree;
  pcl::KdTreeFLANN<pcl::PointXYZRGBA, flann::L2_Simple<float>> kdtreeXY;

public:
  KSearcher();
  virtual ~KSearcher();
  void setInput(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &);
  void setProjectedInput(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &);
  std::vector<int> KSearch(pcl::PointXYZRGBA &,int);
  std::vector<float> CylinderSearch(pcl::PointXYZRGBA &,float);
};

#endif
