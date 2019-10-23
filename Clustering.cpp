#include "Clustering.h"

Clustering::Clustering(){
}

// Set samples
void Clustering::setSamples(cv::Mat & samples){
  trainingData.resize(samples.cols, samples.rows);

  for (int i = 0; i < trainingData.n_cols; ++i) {
      std::vector<double> row = samples.row(i);
      for (int j = 0; j < row.size(); ++j)
        trainingData.at(j,i) = row[j];
    }
}

// KMeans clustering, allows empty clusters
void Clustering::kmeans (int k, int attempts)
{
  arma::Row<size_t> assignments;
  arma::mat centroids;

  mlpack::kmeans::KMeans<mlpack::metric::ManhattanDistance,mlpack::kmeans::SampleInitialization,
      mlpack::kmeans::AllowEmptyClusters, mlpack::kmeans::NaiveKMeans> kmeans(attempts);

  kmeans.Cluster(trainingData, size_t(k), assignments, centroids);
  n_clusters = k;

  clusterSizes.resize(n_clusters);
  clusterIndices.resize(n_clusters);


  // Store centers
  for (int i = 0; i < centroids.n_cols; ++i){
      cv::Mat temp(1, centroids.n_rows, CV_32FC1);

      for (int j = 0; j < centroids.n_rows; ++j)
        temp.at<float>(0,j) = (float)centroids.at(j,i);

      centers.push_back(temp);
    }

  // Store point assigments and increment cluster sizes
  for (int i = 0; i < assignments.n_cols; ++i) {
      labels.push_back(assignments(i));
      clusterSizes[assignments(i)]++;
      clusterIndices[assignments(i)].push_back(i);
    }

  genColors(n_clusters);
}


// KMeans clustering, but don't allow empty clusters
void Clustering::kmeansNoEmpty (int k, int attempts)
{
  arma::Row<size_t> assignments;
  arma::mat centroids;

  mlpack::kmeans::KMeans<mlpack::metric::ManhattanDistance,mlpack::kmeans::SampleInitialization,
      mlpack::kmeans::MaxVarianceNewCluster, mlpack::kmeans::NaiveKMeans> kmeans(attempts);

  kmeans.Cluster(trainingData, size_t(k), assignments, centroids);
  n_clusters = k;

  clusterSizes.resize(n_clusters);
  clusterIndices.resize(n_clusters);

  // Store centers
  for (int i = 0; i < centroids.n_cols; ++i){
      cv::Mat temp(1, centroids.n_rows, CV_32FC1);

      for (int j = 0; j < centroids.n_rows; ++j)
        temp.at<float>(0,j) = (float)centroids.at(j,i);

      centers.push_back(temp);
    }

  // Store point assigments and increment cluster sizes
  for (int i = 0; i < assignments.n_cols; ++i) {
      labels.push_back(assignments(i));
      clusterSizes[assignments(i)]++;
      clusterIndices[assignments(i)].push_back(i);
    }

  genColors(n_clusters);
}


// DBScan clustering
void Clustering::dbscan(double eps, int minPts) {

  arma::Row<size_t> assignments;
  mlpack::range::RangeSearch<> RS;
  mlpack::dbscan::DBSCAN<mlpack::range::RangeSearch<>> dbs(eps, minPts, true, RS);

  n_clusters = (int)dbs.Cluster(trainingData, assignments);

  // Store point assigments
  for (int i = 0; i < assignments.n_cols; ++i)
    labels.push_back(assignments(i));

  genColors(n_clusters);
}

void Clustering::genColors(int n){
  // Use custom values for first 6 colours
  std::vector<int> rgb = {255,0,0}; colors.push_back(rgb);
  rgb = {0,255,0}; colors.push_back(rgb);
  rgb = {0,0,255}; colors.push_back(rgb);
  rgb = {255,255,0}; colors.push_back(rgb);
  rgb = {0,255,255}; colors.push_back(rgb);
  rgb = {255,0,255}; colors.push_back(rgb);

  if (n > 6){
      // Generate random colors for extra clusters
      for (int i = 0; i < n; ++i){
          std::random_device seeder;
          std::ranlux48 gen(seeder());
          std::uniform_int_distribution<int>  uniform_0_255(0, 255);
          rgb = {uniform_0_255(gen), uniform_0_255(gen),uniform_0_255(gen)};
          colors.push_back(rgb);
        }
    }
}

// Colors all scan clusters and saves to file
void Clustering::saveClusters(std::string filename, cv::Mat & superpixelIds, cv::Mat & original, int processed)
{
  cv::Mat painted = original;

  for (int i = 0; i < superpixelIds.cols; ++i)
    for (int j = 0; j < superpixelIds.rows; ++j)
      {
        int id = superpixelIds.at<int>(j,i) + processed;

        if (labels[id] >= 0 && labels[id] != SIZE_MAX) {
            painted.at<cv::Vec3b>(j, i).val[0] = colors[labels[id]][0];
            painted.at<cv::Vec3b>(j, i).val[1] = colors[labels[id]][1];
            painted.at<cv::Vec3b>(j, i).val[2] = colors[labels[id]][2];
          }
      }
  cv::imwrite(filename, painted);
}

void Clustering::clearVecs(){
  labels.clear();
  colors.clear();
  centers.release();
  clusterSizes.clear();
  clusterIndices.clear();
}
