#ifndef CLOUDREADER_H_
#define CLOUDREADER_H_

#include "Common.h"
#include "opencv2/highgui.hpp"
#include <pcl/io/pcd_io.h>

class CloudReader {

public:
  CloudReader();
  virtual ~CloudReader();
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr readToCloud(boost::filesystem::path & file_name);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr readSemanticToCloud(boost::filesystem::path & file_name);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr readKDToCloud(boost::filesystem::path & file_name, std::string);
  void splitCloud(boost::filesystem::path, boost::filesystem::path  file_name);
  cv::Mat readToRGB(const std::string & file_name);
};

#endif
