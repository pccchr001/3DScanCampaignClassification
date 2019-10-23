#ifndef SCANSELECTION_H
#define SCANSELECTION_H

#include "Common.h"
#include "Clustering.h"
#include "CloudFex.h"
#include "Segmenter.h"
#include "pcl/segmentation/supervoxel_clustering.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>

class ScanSelection
{
public:
  ScanSelection();
  std::vector<boost::filesystem::path> supervoxelSelect (int, std::vector<boost::filesystem::path>,int, bool);

private:

};

#endif
