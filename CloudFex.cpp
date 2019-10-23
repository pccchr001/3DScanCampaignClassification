#include "CloudFex.h"

using namespace Eigen;
using namespace boost::assign;

CloudFex::CloudFex(){
}

CloudFex::~CloudFex() {
}

struct str{
  float value;int index;
};

int cmp(const void *a,const void *b)
{
  struct str *a1 = (struct str *)a;
  struct str *a2 = (struct str*)b;
  if((*a1).value>(*a2).value)return -1;
  else if((*a1).value<(*a2).value)return 1;
  else return 0;
}

// Calculates 3D and 2D Shape and Geometric Features
void CloudFex::calcPointFeat (pcl::PointXYZRGBA & example, std::vector<int> & knn,
                              std::vector<float> & cylinder_h, int & label, int & level, std::vector<float> * vec)
{
  float moments[4] = {0,0,0,0};
  VectorXd V[3];

  MatrixXd P(10,3);

  P(0,0) = example.x;
  P(0,1) = example.y;
  P(0,2) = example.z;

  for (u_int x = 1; x < 10 ; ++x)	{
      P(x,0) = points[level](knn[x-1],0);
      P(x,1) = points[level](knn[x-1],1);
      P(x,2) = points[level](knn[x-1],2);
    }

  MatrixXd centered = P.rowwise() - P.colwise().mean();
  MatrixXd cov = (centered.adjoint() * centered) / double(P.rows() - 1);
  ComplexEigenSolver<MatrixXd> es(cov, true);

  float arr[3]={es.eigenvalues().real()(0),es.eigenvalues().real()(1),es.eigenvalues().real()(2)};
  struct str objects[3];

  for (int i = 0; i < 3; ++i){
      objects[i].value=arr[i];
      objects[i].index=i;
    }
  qsort(objects,3,sizeof(objects[0]),cmp);

  float EVs[3];
  for (int i = 0; i < 3; ++i){
      EVs[i] = es.eigenvalues().real()(objects[i].index);
      V[i] = es.eigenvectors().real().col(objects[i].index);
    }

  assert (EVs[0] >=  EVs[1] >= EVs[2] >= 0); // λ1 ≥ λ2 ≥ λ3 ≥ 0

  float sum_EVs;
  sum_EVs = EVs[0] + EVs[1] + EVs[2];

  Matrix<double, 1, 3> A;
  A << 0,0,1;

  float verticality = 1 - fabs(A.dot(V[2]));


  // Moments
  for (int k = 1; k < 10; ++k)
    {
      VectorXd p_diff = P.row(k)- P.row(0);
      moments[0] += p_diff.dot(V[0]); // First order, first axis
      moments[1] += p_diff.dot(V[1]); // First order, second axis
      moments[2] += pow((p_diff.dot(V[0])),2); // Second order, first axis
      moments[3] += pow((p_diff.dot(V[1])),2); // Second order, second axis
    }

  // Height Features // Replaced with non-cylindrical height features
  float vertical_range = 0;
  float height_below = 0;
  float height_above = 0;
  if (cylinder_h.size() == 2){ // If cylindrical neighbourhood has a max and min z
      float z = example.z;
      vertical_range = cylinder_h[1]-cylinder_h[0];
      height_below = z -cylinder_h[0];
      height_above = cylinder_h[1] - z;
    }

  // Normalize EVs
  EVs[0] = EVs[0] / sum_EVs;
  EVs[1] = EVs[1] / sum_EVs;
  EVs[2] = EVs[2] / sum_EVs;

  assert (EVs[0] >= EVs[1] >= EVs[2] >= 0);

  float linearity = ( EVs[0] - EVs[1] ) / EVs[0];
  float planarity = ( EVs[1] - EVs[2] ) / EVs[0];
  float sphericity = EVs[2] / EVs[0];
  float anisotropy = ( EVs[0] - EVs[2] )/ EVs[0];
  float surface_variation = EVs[2] / ( EVs[0] + EVs[1] + EVs[2] );
  float omnivariance = pow(( EVs[0] * EVs[1] * EVs[2] ),(1.0/3.0));

  // Eigenvalues are >= 0, so log(0) might occur
  float eigenentropy = 0;
  for (int i = 0; i < 3 ; ++i)
    if (EVs[i] > 0) // Check that EV is > 0 before log calc
      eigenentropy += EVs[i] * (log(EVs[i]));
  eigenentropy *= -1;

  // 3D Geometric Features

  // distance D 3D between Xi and the farthest point in the local neighborhood
  // Farthest point is the last entry in neighbourhood due to KNN search
  float max_dist = 0;

  for (u_int x = 1; x < 10 ; ++x){
      if  ((P.row(0) - P.row(9)).cwiseAbs().sum() > max_dist)
        max_dist = (P.row(0) - P.row(9)).cwiseAbs().sum();
    }

  float radius3D = max_dist; // estimate radius as this distance

  // the local point density ρ3D
  float density3D = 10/(4/3*M_PI*pow(radius3D, 3.0));

  std::vector<float> ZVals;
  for (u_int x = 0; x < 10 ; ++x){
      ZVals.push_back(P.row(x)[2]); // Push all Z values to vector
    }
  // maximum difference ∆H of the height values within the local neighborhood
  float maxZ = *std::max_element(ZVals.begin(), ZVals.end());
  float minZ = *std::min_element(ZVals.begin(), ZVals.end());
  float deltaH = maxZ - minZ;

  // standard deviation σH of the height values within the local neighborhood
  float mean = std::accumulate( ZVals.begin(), ZVals.end(), 0.0)/ZVals.size();
  float variance = 0;
  for (int x = 0; x < ZVals.size(); x++) {
      variance += (ZVals.at(x) - mean) * (ZVals.at(x) - mean);
    }
  variance /= ZVals.size();
  float stdDH = sqrt(variance);

  // 2D Shape Features
  MatrixXd P2D(10,2);

  P2D(0,0) = example.x;
  P2D(0,1) = example.y;

  for (u_int x = 1; x < 10 ; ++x){
      P2D(x,0) = points[level](knn[x-1],0);
      P2D(x,1) = points[level](knn[x-1],1);
    }

  MatrixXd centered2D = P2D.rowwise() - P2D.colwise().mean();
  MatrixXd cov2D = (centered2D.adjoint() * centered2D) / double(P2D.rows() - 1);
  ComplexEigenSolver<MatrixXd> es2D(cov2D, true);

  float arr2D[2]={es2D.eigenvalues().real()(0),es2D.eigenvalues().real()(1)};
  struct str objects2D[2];

  for (int i = 0; i < 2; ++i){
      objects2D[i].value=arr2D[i];
      objects2D[i].index=i;
    }
  qsort(objects2D,2,sizeof(objects2D[0]),cmp);

  // Sum of 2D Evs of 2D structure tensor (strip away Z from points before calcling eigenvalues)
  float EVs2D[2];
  VectorXd V2D[2];
  for (int i = 0; i < 2; ++i){
      EVs2D[i] = es2D.eigenvalues().real()(objects2D[i].index);
      V2D[i] = es2D.eigenvectors().real().col(objects2D[i].index);
    }

  assert (EVs2D[0] >= EVs2D[1] >= 0);

  float sum_EVs2D;
  sum_EVs2D = EVs2D[0] + EVs2D[1];

  // Ratio of 2D eigenvalues (1/0)
  float EVs2DRatio = EVs2D[1] / EVs2D[0];

  // 2D Geometric Features
  // Distance D2D between Xi and furthest point
  float max_dist2D = 0;

  for (u_int x = 1; x < 10 ; ++x){
      if  ((P2D.row(0) - P2D.row(x)).cwiseAbs().sum() > max_dist2D)
        max_dist2D = (P2D.row(0) - P2D.row(x)).cwiseAbs().sum();
    }

  float radius2D = max_dist2D;

  // Local point Density P2D in 2D space
  float density2D = 10/(M_PI* pow(radius2D,2));

  std::vector<float> temp;
  temp.reserve(24);

  temp +=
      sum_EVs, // 0
      verticality, // 1
      linearity, // 2
      planarity, // 3
      surface_variation, // 4
      sphericity, // 5
      omnivariance, // 6
      anisotropy, // 7
      eigenentropy, // 8
      radius3D, // 9
      density3D, // 10
      deltaH, // 11
      stdDH, // 12
      sum_EVs2D, // 13
      EVs2DRatio, // 14
      radius2D, // 15
      density2D, // 16
      moments[0], // 17
      moments[1], // 18
      moments[2], // 19
      moments[3], // 20
      vertical_range, //21
      height_above, // 22
      height_below; //23

  for (int i = 0; i < temp.size(); ++i){
      vec->at(level*(int)temp.size()+i) = temp[i];
    }

  if (level == max_level-1){
      vec->at(level*temp.size()+temp.size()) = label ;
    }
}

// Extract supervoxel-based features
void CloudFex::calcVoxelFeat(pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr & supervoxel, std::vector<int> cylinder_indices){

  int voxelSize = supervoxel->voxels_->points.size();

  pcl::PointXYZRGBA centroid;
  supervoxel->getCentroidPoint(centroid); // Store supervoxel centroid

  MatrixXd CentroidP(1,3);
  CentroidP(0,0) = centroid.x;
  CentroidP(0,1) = centroid.y;
  CentroidP(0,2) = centroid.z;

  float moments[4] = {0,0,0,0};
  VectorXd V[3];

  MatrixXd P(voxelSize,3);

  for (u_int x = 0; x < voxelSize ; ++x){
      P(x,0) = supervoxel->voxels_->points.at(x).x;
      P(x,1) = supervoxel->voxels_->points.at(x).y;
      P(x,2) = supervoxel->voxels_->points.at(x).z;
    }

  MatrixXd centered = P.rowwise() - P.colwise().mean();
  MatrixXd cov = (centered.adjoint() * centered) / double(P.rows() - 1);
  ComplexEigenSolver<MatrixXd> es(cov, true);

  float arr[3]={es.eigenvalues().real()(0),es.eigenvalues().real()(1),es.eigenvalues().real()(2)};
  struct str objects[3];

  for (int i = 0; i < 3; ++i){
      objects[i].value=arr[i];
      objects[i].index=i;
    }
  qsort(objects,3,sizeof(objects[0]),cmp);

  float EVs[3];
  for (int i = 0; i < 3; ++i){
      EVs[i] = es.eigenvalues().real()(objects[i].index);
      V[i] = es.eigenvectors().real().col(objects[i].index);
    }

  float sum_EVs;
  sum_EVs = EVs[0] + EVs[1] + EVs[2];

  Matrix<double, 1, 3> A;
  A << 0,0,1;

  float verticality = 1 - fabs(A.dot(V[2]));


  // Moments
  for (int k = 0; k < voxelSize; ++k)
    {
      VectorXd p_diff = P.row(k)- CentroidP.row(0);
      moments[0] += p_diff.dot(V[0]); // First order, first axis
      moments[1] += p_diff.dot(V[1]); // First order, second axis
      moments[2] += pow((p_diff.dot(V[0])),2); // Second order, first axis
      moments[3] += pow((p_diff.dot(V[1])),2); // Second order, second axis
    }

  // Height Features based on cylindrical neighbourhood found in centroid cloud
  float max, min, vertical_range, height_below, height_above;
  max = min = vertical_range = height_below = height_above =0;

  float z = centroid.z;
  max = min = z;

  for (int i = 0; i < cylinder_indices.size(); ++i){
      float tempZ = centroid_cloud->points.at(cylinder_indices[i]).z;
      if (tempZ > max)
        max = tempZ;
      else if (tempZ < min)
        min = tempZ;
    }

  vertical_range = max - min;
  height_below = z -min;
  height_above = max - z;


  // Normalize EVs
  EVs[0] = EVs[0] / sum_EVs;
  EVs[1] = EVs[1] / sum_EVs;
  EVs[2] = EVs[2] / sum_EVs;

  assert (EVs[0] >= EVs[1] >= EVs[2] >= 0);

  float linearity = ( EVs[0] - EVs[1] ) / EVs[0];
  float planarity = ( EVs[1] - EVs[2] ) / EVs[0];
  float sphericity = EVs[2] / EVs[0];
  float anisotropy = ( EVs[0] - EVs[2] )/ EVs[0];
  float surface_variation = EVs[2] / ( EVs[0] + EVs[1] + EVs[2] );

  // Eigenvalues are >= 0, so log(0) might occur
  float eigenentropy = 0;
  for (int i = 0; i < 3 ; ++i)
    if (EVs[i] > 0) // Check that EV is > 0 before log calc
      eigenentropy += EVs[i] * (log(EVs[i]));
  eigenentropy *= -1;

  // 3D Geometric Features
  // distance D 3D between centroid and the farthest point in the supervoxel
  // Cannot rely on KNN results so calculate max distance manually
  float radius3D = 0;
  for (int i = 0; i < voxelSize; ++i){
      float manh_dist = (CentroidP.row(0) - P.row(i)).cwiseAbs().sum();
      if (manh_dist > radius3D)
        {radius3D = manh_dist;}
    }

  // the density of the supervoxel
  float density3D = voxelSize/((4/3)*M_PI*pow(radius3D, 3.0));

  // Store Z vals and calculate average height of supervoxel
  std::vector<float> ZVals;
  float height = 0;

  for (u_int x = 0; x < voxelSize ; ++x){
      ZVals.push_back(P.row(x)[2]); // Push all Z values to vector
      height += P.row(x)[2];
    }
  height /= voxelSize;

  // maximum difference ∆H of the height values within the supervoxel
  float maxZ = *std::max_element(ZVals.begin(), ZVals.end());
  float minZ = *std::min_element(ZVals.begin(), ZVals.end());
  float deltaH = maxZ - minZ;

  // standard deviation σH of the height values within the supervoxel
  float mean = std::accumulate( ZVals.begin(), ZVals.end(), 0.0)/ZVals.size();
  float varianceH = 0;
  for (int x = 0; x < ZVals.size(); x++) {
      varianceH += (ZVals.at(x) - mean) * (ZVals.at(x) - mean);
    }
  varianceH /= ZVals.size();
  float stdDH = sqrt(varianceH);

  // 2D Shape Features
  MatrixXd P2D(voxelSize,2);
  MatrixXd CentroidP2D(1,2);
  CentroidP2D(0,0) = centroid.x;
  CentroidP2D(0,1) = centroid.y;

  for (u_int x = 0; x < voxelSize ; ++x){
      P2D(x,0) = supervoxel->voxels_->points.at(x).x;
      P2D(x,1) = supervoxel->voxels_->points.at(x).y;
    }

  MatrixXd centered2D = P2D.rowwise() - P2D.colwise().mean();
  MatrixXd cov2D = (centered2D.adjoint() * centered2D) / double(P2D.rows() - 1);
  ComplexEigenSolver<MatrixXd> es2D(cov2D, true);

  float arr2D[2]={es2D.eigenvalues().real()(0),es2D.eigenvalues().real()(1)};
  struct str objects2D[2];

  for (int i = 0; i < 2; ++i){
      objects2D[i].value=arr2D[i];
      objects2D[i].index=i;
    }
  qsort(objects2D,2,sizeof(objects2D[0]),cmp);

  // Sum of 2D Evs of 2D structure tensor (strip away Z from points before calcling eigenvalues)
  float EVs2D[2];
  VectorXd V2D[2];
  for (int i = 0; i < 2; ++i){
      EVs2D[i] = es2D.eigenvalues().real()(objects2D[i].index);
      V2D[i] = es2D.eigenvectors().real().col(objects2D[i].index);
    }

  assert (EVs2D[0] >= EVs2D[1] >= 0);

  float sum_EVs2D;
  sum_EVs2D = EVs2D[0] + EVs2D[1];

  // Ratio of 2D eigenvalues (1/0)
  float EVs2DRatio = EVs2D[1] / EVs2D[0];

  // 2D Geometric Features
  // Distance D2D between centroid and furthest point in supervoxel
  float radius2D = 0;
  for (int i = 0; i < voxelSize; ++i){
      float manh_dist = (CentroidP2D.row(0) - P2D.row(i)).cwiseAbs().sum();
      if (manh_dist > radius2D)
        {radius2D = manh_dist;}
    }

  // Features derived directly from supervoxel centroid
  pcl::PointNormal centroidnorm;
  supervoxel->getCentroidPointNormal(centroidnorm);
  float centroidNormZ = centroidnorm.normal_z; // X and Y norms not useful

  features +=
      verticality,
      linearity,
      planarity,
      surface_variation,
      sphericity,
      anisotropy,
      deltaH,
      stdDH,
      sum_EVs2D,
      EVs2DRatio ,
      centroidNormZ,
      voxelSize;
}

void CloudFex::setPointMatrix(Eigen::MatrixXd p[], int max_lev){
  points = p;
  max_level = max_lev;
}

void CloudFex::setCentroidCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud){
  centroid_cloud = cloud;
}

void CloudFex::saveFeatures(std::string filename){
  std::ofstream ofile(filename.c_str(),std::ofstream::app);

  for (size_t i = 0; i < features.size(); i++)
    ofile << features[i] << ",";

  ofile << "\n";
  ofile.close();
}

void CloudFex::clearFeatures(){
  features.clear();
}

std::vector<float> CloudFex::getFeatures(){
  return features;
}
