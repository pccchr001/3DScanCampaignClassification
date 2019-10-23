#include "Segmenter.h"

Segmenter::Segmenter ()
{
}

Segmenter::~Segmenter ()
{
}

void Segmenter::setInput(cv::Mat & lab_){
  lab = lab_;
  width = lab.size().width;
  height = lab.size().height;
}

// SEEDS segmentation
void Segmenter::segmentSEEDS(){

  //  int num_iterations = 100;
  //  int prior = 2;
  //  bool double_step = false;
  //  int num_superpixels = 5000;
  //  int num_levels = 4;
  //  int num_histogram_bins = 5;

  //  seeds = cv::ximgproc::createSuperpixelSEEDS(width, height, lab.channels(), num_superpixels,
  //                                              num_levels, prior, num_histogram_bins, double_step);
  //  seeds->iterate(lab, num_iterations);

  //  seeds->getLabelContourMask(mask, false);
  //  num_superpixels = seeds->getNumberOfSuperpixels();
  //  seeds->getLabels(labels);
}

// SLIC segmentation
void Segmenter::segmentSLIC(){

  //  int region_size = 12;
  //  float ruler = 10;
  //  int min_element_size = 12;
  //  int num_iterations = 5;

  //  slic = cv::ximgproc::createSuperpixelSLIC(lab,101,region_size,ruler);
  //  slic->iterate(num_iterations);

  //  if (min_element_size>0)
  //    slic->enforceLabelConnectivity(min_element_size);

  //  slic->getLabelContourMask(mask, true);
  //  num_superpixels = slic->getNumberOfSuperpixels();
  //  slic->getLabels(labels);
}

// LSC segmentation
void Segmenter::segmentLSC(){

  //  int region_size = 20;
  //  float ratio = 0.075f;
  //  int min_element_size = 10;
  //  int num_iterations = 10;

  //  lsc = cv::ximgproc::createSuperpixelLSC(lab, region_size, ratio);
  //  lsc->iterate(num_iterations);

  //  if (min_element_size>0)
  //    lsc->enforceLabelConnectivity(min_element_size);

  //  lsc->getLabelContourMask(mask, true);
  //  num_superpixels = lsc->getNumberOfSuperpixels();
  //  lsc->getLabels(labels);
}

std::map <uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> Segmenter::segmentVCCS(float seed_resolution, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &inputCloud){

  bool single_transform = false;
  float voxel_resolution = 0.075f;

  float color_importance = 0.0f;    // Don't use colour
  float spatial_importance = 0.3f; // Spatial regularity (higher = more regular, lower = irregular but will respect color & normals more)
  float normal_importance = 1.0f; // Keep this high for e.g. where ground meets walls

  pcl::SupervoxelClustering<pcl::PointXYZRGBA> super (voxel_resolution, seed_resolution);

  super.setUseSingleCameraTransform (single_transform);
  super.setColorImportance (color_importance);
  super.setSpatialImportance (spatial_importance);
  super.setNormalImportance (normal_importance);
  super.setInputCloud(inputCloud);

  std::map <uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> supervoxels;
  super.extract (supervoxels);

  labelcloud = super.getLabeledCloud();
  centroidcloud = super.getVoxelCentroidCloud();
  centroidcloud2D = super.getVoxelCentroidCloud();

  for (int i = 0; i < centroidcloud2D->points.size(); ++i){
      centroidcloud2D->points.at(i).z = 0; // strip Z data from centroid cloud = projected to XY plane
    }

  return supervoxels;
}

cv::Mat& Segmenter::getLabelMat(){
  return labels;
}

int Segmenter::getSuperpixelCount(){
  return num_superpixels;
}

cv::Mat& Segmenter::getResultMask(){
  return mask;
}

pcl::PointCloud<pcl::PointXYZL>::Ptr Segmenter::getLabelledCloud(){
  return labelcloud;
}
