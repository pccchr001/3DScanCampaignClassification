#include "ImgFex.h"

ImgFex::ImgFex (int n_superpixels_, bool lite_)
{
  ns = n_superpixels_;
  lite = lite_;
}

ImgFex::~ImgFex ()
{
}

void ImgFex::setImages(cv::Mat & rgb_, cv::Mat & lab_, cv::Mat & label){
  rgb = rgb_;
  lab = lab_;
  labels = label;
  superpixel_feat.resize(ns);
  if (!lite){
      cv::copyMakeBorder(rgb_, ctx_rgb, 10, 10, 10, 10, cv::BORDER_REFLECT);
      cv::copyMakeBorder(lab_, ctx_lab, 10, 10, 10, 10, cv::BORDER_REFLECT);
      cv::copyMakeBorder(label, ctx_labels, 10, 10, 10, 10, cv::BORDER_REFLECT);}
}

// Calculate superpixel features
void ImgFex::calcSuperpixelFeat(){

  float minX [ns], maxX[ns], minY[ns], maxY[ns], meanX[ns], meanY[ns],count[ns];

  for (int i = 0; i < ns; ++i){
      count[i] = 0;
    }

  // Determine bounding box, count and mean position
  for (int y = 0; y < labels.rows; ++y){
      for (int x = 0; x < labels.cols; ++x){

          int label = labels.at<int>(y,x);

          // Position
          meanX[label] += x;
          meanY[label] += y;

          if (count[label] == 0) {
              minX[label] = maxX[label] = x;
              minY[label] = maxY[label] = y;
            }
          else {
              if (x < minX[label])
                minX[label] = x;
              if (x > maxX[label])
                maxX[label] = x;
              if (y < minY[label])
                minY[label] = y;
              if (y > maxY[label])
                maxY[label] = y;
            }
          count[label]++;
        }
    }

  for (int i = 0; i < ns; ++i){
      meanX[i] /= count[i];
      meanY[i] /= count[i];
    }

  // For each superpixel
  for (int i = 0; i < ns; ++i){

      cv::Rect rec(cv::Point(minX[i],minY[i]), cv::Point(maxX[i],maxY[i]));
      cv::Mat mask = labels(rec) == i;

      // Position
      superpixel_feat[i].push_back(meanX[i]); // Mean X
      superpixel_feat[i].push_back(meanY[i]); // Mean Y

      // Colour Features
      cv::Scalar meanRGB,stdRGB, meanLAB, stdLAB;

      cv::meanStdDev(rgb(rec), meanRGB,stdRGB,mask);
      cv::meanStdDev(lab(rec), meanLAB,stdLAB,mask);

      for (int j = 0; j < 3; ++j){
          superpixel_feat[i].push_back(meanRGB[j]);
          superpixel_feat[i].push_back(stdRGB[j]);
          superpixel_feat[i].push_back(meanLAB[j]);
          superpixel_feat[i].push_back(stdLAB[j]);
        }


      // Calculate more complex features
      if (!lite){

          // Get superpixel contours
          std::vector<std::vector<cv::Point> > contours;
          cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

          // Get idx of biggest contour
          int idx = 0;
          if (contours.size() > 1)
            for (u_int j = 1; j < contours.size(); j++)
              if (contours[j].size() > contours[idx].size())
                idx = j;

          // Fit ellipse to biggest contour
          cv::Mat points;
          cv::Mat(contours[idx]).convertTo(points, CV_32F);
          cv::RotatedRect ellipse = fitEllipse(points);

          // Size/Shape features
          superpixel_feat[i].push_back(M_PI*(ellipse.size.height/2)*(ellipse.size.width/2)); // Area
          superpixel_feat[i].push_back(sqrt(4*ellipse.size.area()/M_PI)); // Equivalent diameter
          superpixel_feat[i].push_back(ellipse.size.height); // Major axis
          superpixel_feat[i].push_back(ellipse.size.width); // Minor axis
          superpixel_feat[i].push_back(ellipse.angle); // Orientation
          superpixel_feat[i].push_back(ellipse.size.height/ellipse.size.width); // Eccentricity

          // Superpixel mask features
          cv::Mat resized;
          cv::resize(mask, resized, cv::Size(8, 8),0,0, cv::INTER_NEAREST); // 64 pixels of val 0 or 255
          for(int y = 0; y < resized.rows; ++y)
            for(int x = 0; x < resized.cols; ++x)
              superpixel_feat[i].push_back((int)resized.at<uchar>(y,x)/255); // Superpixel mask 0 or 1 (8x8)

          // Colour histogram features
          std::vector<float> histo_feat = getHistograms(rec, mask,false);
          for (u_int h = 0; h < histo_feat.size(); ++h)
            superpixel_feat[i].push_back(histo_feat.at(h)); // Colour histograms

          // Contextual colour features
          cv::Rect ctx_rec(cv::Point(minX[i]+10-10,minY[i]+10-10),
                           cv::Point(maxX[i]+10+10,maxY[i]+10+10)); // Add 10 to get original idx, pad 10 to get ctx idx
          cv::Mat ctx_mask;
          cv::bitwise_not(ctx_labels(ctx_rec) == i, ctx_mask);

          cv::Scalar meanRGBctx,stdRGBctx, meanLABctx, stdLABctx;
          cv::meanStdDev(ctx_rgb(ctx_rec), meanRGBctx,stdRGBctx);
          cv::meanStdDev(ctx_lab(ctx_rec), meanLABctx,stdLABctx);

          for (int j = 0; j < 3; ++j){
              superpixel_feat[i].push_back(meanRGBctx[j]);
              superpixel_feat[i].push_back(stdRGBctx[j]);
              superpixel_feat[i].push_back(meanLABctx[j]);
              superpixel_feat[i].push_back(stdLABctx[j]);
            }

          // Contextual colour histogram features
          std::vector<float> ctx_histo_feat = getHistograms(ctx_rec, ctx_mask,true);
          for (u_int h = 0; h < ctx_histo_feat.size(); ++h)
            superpixel_feat[i].push_back(ctx_histo_feat.at(h));
        }
    }
}

// Get histograms method
std::vector<float> ImgFex::getHistograms(cv::Rect rec, cv::Mat mask, bool ctx)
{
  std::vector<cv::Mat> bgr_planes, lab_planes;
  if (!ctx){
      cv::split(rgb(rec), bgr_planes);
      cv::split(lab(rec), lab_planes);
    }
  else{
      cv::split(ctx_rgb(rec), bgr_planes);
      cv::split(ctx_lab(rec), lab_planes);
    }

  bgr_planes.insert(bgr_planes.end(), lab_planes.begin(), lab_planes.end());

  int histSize = 8;
  float range[] = {0, 256}; // B,G,R and L,a,b (Conversion sets L,a,b to this range)
  const float* histRange = {range};
  std::vector<float> histo_feat;

  for (int h = 0; h < 6; ++h){
      cv::Mat histogram;
      cv::calcHist( &bgr_planes[h], 1, 0, mask, histogram, 1, &histSize, &histRange);
      for (int x = 0; x < 8; ++x)
        histo_feat.push_back(histogram.at<float>(x));
    }
  return histo_feat;
}

// Get superpixel features for a superpixel
std::vector<float> ImgFex::getSuperpixelFeat(int idx){
  return superpixel_feat.at(idx);
}
