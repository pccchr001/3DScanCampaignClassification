#include "KSearcher.h"

KSearcher::KSearcher (){
}

KSearcher::~KSearcher() {
}

// Set cloud as KDTree's search space
void KSearcher::setInput(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud){
  kdtree.setInputCloud(cloud);
}

// Projects cloud onto XY plane and set it as XY KDtree's search space
void KSearcher::setProjectedInput(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud){

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr projected (new pcl::PointCloud<pcl::PointXYZRGBA>);

  for (size_t i = 0; i < cloud->size(); i++){
    pcl::PointXYZRGBA p = cloud->at(i);
    p.z = 0;
    projected->push_back(p);
  }
  kdtreeXY.setInputCloud(projected);
}

// Finds (up to) K nearest neighbours to a point
std::vector<int>KSearcher::KSearch(pcl::PointXYZRGBA &point, int k){

  std::vector<int> knn;
  std::vector<int> indices(k);
  std::vector<float> distances(k);

  if (kdtree.nearestKSearch (point, k, indices, distances) > 0 ){
    for (size_t i = 0; i < indices.size (); ++i){
      knn.push_back(indices[i]);
    }
  }
  return knn;
}

// Finds minimum and maximum Z values of point's neighbours in cylinder with radius R
std::vector<float> KSearcher::CylinderSearch(pcl::PointXYZRGBA &point, float r){

  std::vector<int> indices;
  std::vector<float> distances;
  std::vector<float> minmax_z;
  pcl::PointXYZRGBA searchPoint = point;
  searchPoint.z = 0;

  if (kdtreeXY.radiusSearch (searchPoint, r, indices, distances) > 0 ){
    float min_z, max_z;
    min_z = max_z = kdtree.getInputCloud()->at(indices[0]).z;

    for (size_t i = 1; i < indices.size() ; ++i){
      float z = kdtree.getInputCloud()->at(indices[i]).z;
      if (z > max_z)
        max_z = z;
      else if (z < min_z)
        min_z = z;
    }

    minmax_z.push_back(min_z);
    minmax_z.push_back(max_z);
}

  return minmax_z;
}
