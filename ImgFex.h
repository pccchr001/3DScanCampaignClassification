#ifndef ImgFex_H_
#define ImgFex_H_

#include "Common.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/features2d.hpp>

class ImgFex
{
public:
  ImgFex (int, bool);
  virtual ~ImgFex ();
  void setImages(cv::Mat &, cv::Mat &,cv::Mat &);
  void calcSuperpixelFeat();
  std::vector<float> getSuperpixelFeat(int);

private:

  int ns;
  bool lite;
  cv::Mat rgb;
  cv::Mat lab;
  cv::Mat labels;
  cv::Mat ctx_rgb;
  cv::Mat ctx_lab;
  cv::Mat ctx_labels;
  std::vector<std::vector<float>> superpixel_feat;
  std::vector<float> getHistograms(cv::Rect,cv::Mat,bool);
};

#endif
