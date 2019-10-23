#ifndef VISUALISER_H_
#define VISUALISER_H_

#include "Common.h"
#include <pcl/io/pcd_io.h>
#include <fstream>

class Visualiser{

private:
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr viscloud;
  bool startedWriting = false;
  std::vector<std::vector<int>> colours;

public:
  Visualiser ();
  virtual
  ~Visualiser ();
  void setTestCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &);
  void labelPoint(int,int);
  void visFeature(int ,float);
  void savePredictionCloud(std::string, std::string);
  void saveErrorCloud(std::string,std::string, std::string);
};

#endif
