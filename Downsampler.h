#ifndef DOWNSAMPLER_H_
#define DOWNSAMPLER_H_

#include "Common.h"
#include <fstream>
#include "pcl/octree/octree.h"
#include "pcl/octree/octree_container.h"
#include "pcl/octree/octree_impl.h"
#include "pcl/kdtree/impl/kdtree_flann.hpp"
#include <pcl/io/pcd_io.h>

class Downsampler {
public:
  Downsampler(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &);
  virtual ~Downsampler();
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsample(float);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampleWithLabels (float res, int, boost::filesystem::path, boost::filesystem::path);

private:
  pcl::KdTreeFLANN<pcl::PointXYZRGBA, flann::L2_Simple<float>> kdtree;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
};

#endif
