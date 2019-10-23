#include "Downsampler.h"

Downsampler::Downsampler (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr & cloud_){
  cloud = cloud_;
}

Downsampler::~Downsampler ()
{
}

// Downsamples cloud to given resolution
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr Downsampler::downsample (float res)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampled_cloud (new pcl::PointCloud<pcl::PointXYZRGBA> ());

  pcl::octree::OctreePointCloud<pcl::PointXYZRGBA> octree (res);
  octree.setInputCloud (cloud);
  octree.addPointsFromInputCloud ();
  downsampled_cloud->points.resize(octree.getLeafCount());

  pcl::octree::OctreePointCloud<pcl::PointXYZRGBA>::LeafNodeIterator start;
  pcl::octree::OctreePointCloud<pcl::PointXYZRGBA>::LeafNodeIterator end = octree.leaf_end ();
  int cloudidx = 0;

  for (start = octree.leaf_begin (); start != end; ++start)
    {
      std::vector<int> & indices = start.getLeafContainer ().getPointIndicesVector ();

      float x, y, z, r, g, b;
      x = y = z = r = g = b =  0;

      for (int idx : indices){
          x += (*cloud)[idx].x;
          y += (*cloud)[idx].y;
          z += (*cloud)[idx].z;
          r += (*cloud)[idx].r;
          g += (*cloud)[idx].g;
          b += (*cloud)[idx].b;
        }

      // Get mean XYZIRGB
      x /= indices.size ();
      y /= indices.size ();
      z /= indices.size ();
      r /= indices.size ();
      g /= indices.size ();
      b /= indices.size ();

      // Create new point and assign XYZIRGB
      pcl::PointXYZRGBA p;
      p.x = x;
      p.y = y;
      p.z = z;
      p.r = r;
      p.g = g;
      p.b = b;

      downsampled_cloud->points.at(cloudidx) = p;
      cloudidx++;
    }
  return downsampled_cloud;
}

// Downsamples scan to given resolution and stores labels in vector
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr Downsampler::downsampleWithLabels (float res, int n_classes, boost::filesystem::path labelsIn, boost::filesystem::path labelsOut)
{
  std::vector<int> labels;
  std::ifstream ifile(labelsIn.string());
  int label;
  while (ifile >> label){
      if (label >=1 && label <= n_classes)
        labels.push_back(label-1);
      else
        labels.push_back(99);
    }
  ifile.close();

  std::cout << "labelsize: " << labels.size() << std::endl;
  std::cout << "cloudsize: " << cloud->points.size() << std::endl;

  std::ofstream ofile(labelsOut.string());

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampled_cloud (new pcl::PointCloud<pcl::PointXYZRGBA> ());

  pcl::octree::OctreePointCloud<pcl::PointXYZRGBA> octree (res);
  octree.setInputCloud (cloud);
  octree.addPointsFromInputCloud ();

  pcl::octree::OctreePointCloud<pcl::PointXYZRGBA>::LeafNodeIterator start;
  pcl::octree::OctreePointCloud<pcl::PointXYZRGBA>::LeafNodeIterator end = octree.leaf_end ();

  for (start = octree.leaf_begin (); start != end; ++start)
    {
      std::vector<int> & indices = start.getLeafContainer ().getPointIndicesVector ();

      float x, y, z, intensity, r, g, b, rgb;
      x = y = z = intensity = r = g = b= rgb=  0;
      int label_freq[n_classes];

      for (int i = 0; i < n_classes; ++i)
        label_freq[i] = 0;

      for (int idx : indices)
        {
          x += (*cloud)[idx].x;
          y += (*cloud)[idx].y;
          z += (*cloud)[idx].z;
          r += (*cloud)[idx].r;
          g += (*cloud)[idx].g;
          b += (*cloud)[idx].b;

          if ( labels.at(idx) >= 0 && labels.at(idx) != 99)
            label_freq[labels.at(idx)]++;
        }

      // Get means
      x /= indices.size ();
      y /= indices.size ();
      z /= indices.size ();
      r /= indices.size ();
      g /= indices.size ();
      b /= indices.size ();

      // Create new point and assign XYZIRGB and Label
      pcl::PointXYZRGBA p;
      p.x = x;
      p.y = y;
      p.z = z;
      p.r = r;
      p.g = g;
      p.b = b;

      int max = label_freq[0];
      int maxidx = 0;

      for (int k = 1; k < n_classes; ++k){
          if (label_freq[k] > max){
              max = label_freq[k];
              maxidx = k;
            }
        }

      // push the most frequent point but do not push unlabelled points
      // (shouldn't be used for training & can't be evaluated properly after prediction)
      if (max > 0){
          ofile << maxidx << "\n";
          downsampled_cloud->points.push_back(p);
        }
    }


  ofile.close();
  return downsampled_cloud;
}
