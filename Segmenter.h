#ifndef SEGMENTER_H_
#define SEGMENTER_H_

#include "Common.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>

#include "pcl/segmentation/supervoxel_clustering.h"

class Segmenter
{
public:
  Segmenter ();
  virtual
  ~Segmenter ();
  void setInput(cv::Mat &);
  void segmentSEEDS();
  void segmentSLIC();
  void segmentLSC();
  int getSuperpixelCount();
  cv::Mat& getLabelMat();
  cv::Mat& getResultMask();
  pcl::PointCloud<pcl::PointXYZL>::Ptr getLabelledCloud();
  pcl::PointCloud<pcl::PointXYZ>::Ptr getCentroidCylinder(pcl::PointXYZRGBA);

  //VCCS
  std::map <uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> segmentVCCS(float,pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &);
  pcl::PointCloud<pcl::PointXYZL>::Ptr labelcloud;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr centroidcloud;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr centroidcloud2D;

private:
  int width, height, num_superpixels;
  cv::Mat lab;
  cv::Mat labels;
  cv::Mat mask;
  cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds;
  cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic;
  cv::Ptr<cv::ximgproc::SuperpixelLSC> lsc;
  std::chrono::duration<double> segmentationTime;
};

#endif
