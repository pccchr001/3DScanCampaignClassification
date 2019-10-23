#include "ScanSelection.h"

ScanSelection::ScanSelection(){
}

double cosSimilarity(cv::Mat1f vectorA, cv::Mat1f vectorB)
{
  double adotb = vectorA.dot(vectorB);
  double anorm = sqrt(vectorA.dot(vectorA));
  double bnorm = sqrt(vectorB.dot(vectorB));
  return adotb/(anorm * bnorm);
}


std::vector<boost::filesystem::path> ScanSelection::supervoxelSelect(int n, std::vector<boost::filesystem::path> scans, int selectionScheme, bool multiscale){

  // Selection of scans
  std::vector<boost::filesystem::path> selected;

  if (selectionScheme == 0){

      std::vector<boost::filesystem::path> temp = scans;

      srand (time(NULL));

      for (int i = 0; i < n; ++i){
          int randIdx = rand() % (temp.size());
          std::cout << "random scan: " << temp[randIdx].stem() << std::endl;
          selected.push_back(temp[randIdx]);
          temp.erase(temp.begin() + randIdx);
          std::cout << temp.size() << std::endl;
        }
      return selected;
    }


  int levels = 1;
  if (multiscale) levels = 3;
  assert (n <= scans.size());
  float balanceGoal;
  cv::Mat storedCenters;

  std::vector<cv::Mat> samples(levels);
  std::vector<std::vector<std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr>>> supervoxel_maps(levels);
  std::vector<std::vector<int>> ids[levels];

  for (int j = 0; j < levels; ++j)
    ids[j].resize(scans.size());

  std::cout << "Selecting " << n << " scans from a total of " << scans.size() << " candidates." << std::endl;

  std::chrono::duration<double> voxelFexTime;
  std::chrono::duration<double> segmentationTime;

  for (int i = 0; i < scans.size(); ++i){

      pcl::PCDReader reader;
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA> ());
      cout << "Reading in " << scans[i] << "..." << endl;
      reader.read (scans[i].string(), *cloud);

      // VCCS Segmentation
      float seed_resolution = 0.15f; // Goal size of supervoxels, depends on dataset density

      int seed_multiplier = 2;

      for (int l = 0; l < levels; ++l)
        {
          Segmenter segmenter;
          CloudFex fex;
          std::cout << "Extracting supervoxels of size ~ " << seed_resolution << "..." << std::flush;

          auto segmentationStart = std::chrono::high_resolution_clock::now();
          std::map <uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> supervoxels = segmenter.segmentVCCS(seed_resolution, cloud);
          segmentationTime += std::chrono::high_resolution_clock::now() - segmentationStart;


          std::cout << "Found " << supervoxels.size () << " supervoxels" << std::endl;

          //Visualise segmentation
          pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_voxel_cloud = segmenter.getLabelledCloud();
          labeled_voxel_cloud->width = 1;
          labeled_voxel_cloud->height = labeled_voxel_cloud->points.size();
          pcl::io::savePCDFileASCII ("/media/chris/Seagate Backup Plus Drive/results/clustering/" + scans[i].stem().string() +std::to_string(l) +  "_segments.pcd", *labeled_voxel_cloud);

          std::cout << "Extracting features from supervoxels..." << std::flush;

          auto voxelFexStart = std::chrono::high_resolution_clock::now();
          pcl::KdTreeFLANN<pcl::PointXYZRGBA, flann::L2_Simple<float>> kd;
          kd.setInputCloud(segmenter.centroidcloud2D);
          fex.setCentroidCloud(segmenter.centroidcloud);

          std::map <uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> :: iterator itr;
          for (itr = supervoxels.begin(); itr != supervoxels.end(); ++itr){

              if (itr->second->voxels_->points.size() >= 5){ // If supervoxel has atleast 5 points

                  // Get cylindrical neighbourhood from centroid cloud
                  std::vector<int> cyl_indices;
                  std::vector<float> distances;
                  float r = 0.125f;
                  pcl::PointXYZRGBA searchPoint;
                  itr->second->getCentroidPoint(searchPoint);
                  searchPoint.z = 0;
                  kd.radiusSearch(searchPoint, r, cyl_indices, distances); // Search radius in projected space = cylinder search

                  fex.calcVoxelFeat(itr->second, cyl_indices); // Calc supervoxel feat
                  std::vector<float> voxelfeat = fex.getFeatures(); // Store features at this scale

                  fex.clearFeatures();

                  // Store aggregated features for clustering
                  cv::Mat feat(1,voxelfeat.size(),CV_32FC1);
                  for (int k = 0; k <voxelfeat.size(); ++k)
                    feat.at<float>(0,k) = voxelfeat[k];

                  samples[l].push_back(feat);

                  ids[l][i].push_back(itr->first);
                }
            }
          voxelFexTime += std::chrono::high_resolution_clock::now() - voxelFexStart;
          std::cout << "Done " << std::endl;
          supervoxel_maps[l].push_back(supervoxels);
          seed_resolution *= seed_multiplier; // For next round
        }
    }

  std::cout << "VoxelFex took " << voxelFexTime.count() << "s" << std::endl;
  std::cout << "Segmentation took " << segmentationTime.count() << "s" << std::endl;

  // Perform K means clustering
  std::vector<std::vector<float>> allDistr;
  std::vector<std::vector<float>> allSim;

  for (int l = 0; l < levels; ++l){

      // Standardize samples: x' = (x-mean)/std
      cv::Scalar mean,std;
      for (int i = 0; i < samples[l].cols ; ++i){
          cv::meanStdDev(samples[l].col(i), mean,std);
          for (int j = 0; j < samples[l].col(i).rows; ++j)
            samples[l].at<float>(j,i) = (samples[l].at<float>(j,i)-mean[0])/std[0];
        }

      // -- Start clustering ---
      Clustering clusterer;
      clusterer.setSamples(samples[l]);

      std::cout << "Number of samples to cluster: " << samples[l].size() << std::endl;
      int maxK = 10;
      int attempts = 0;

      // First, perform elbow method to determine optimal K

      auto clusteringStart = std::chrono::high_resolution_clock::now();

      std::cout << "Calculating SSE for K = 1 to K = " << maxK << "..." << std::endl;
      std::vector<float> sse;
      for (int i = 1; i <= maxK ; ++i){

          clusterer.kmeans(i,attempts);
          float kScore = 0;

          for (int j = 0; j < samples[l].rows ; ++j){
              int label = clusterer.labels[j];
              cv::Mat center = clusterer.centers.row(label);
              float dist = cv::norm(samples[l].row(j),center,cv::NORM_L2);
              kScore+=dist;
            }
          sse.push_back(kScore);
          clusterer.clearVecs(); // clear stored assignments
        }

      auto clusteringEnd = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> clusteringTime = clusteringEnd - clusteringStart;
      std::cout << "Clustering took " << clusteringTime.count() << "s" << std::endl;

      std::cout << "SSE for K=1 to K = " << maxK << ":" << std::endl;
      for (u_int i = 0; i < sse.size() ; ++i){
          std::cout << "K = " <<  i+1 << ": " <<  sse[i] << std::endl;
        }

      int k;
      std::cout << "Please input K at the elbow point..." << std::endl;
      std::cin >> k;

      balanceGoal = (float)100/k;
      storedCenters = clusterer.centers;

      std::cout << "Performing KMeans cluster analysis..." << std::endl;
      clusterer.kmeans(k,attempts); // KMeans

      std::cout << "Found clusters: " << clusterer.n_clusters << std::endl;

      // Visualise clustering
      std::cout << "Visualizing supervoxels..." << std::flush;
      int processed = 0;
      for (int s = 0; s < supervoxel_maps[l].size(); ++s){
          pcl::PointCloud<pcl::PointXYZRGBA>::Ptr viscloud (new pcl::PointCloud<pcl::PointXYZRGBA> ());

          for (int k = 0; k < ids[l][s].size() ; ++k){
              pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr temp = supervoxel_maps[l][s][ids[l][s][k]];

              int label = clusterer.labels[k+processed];
              if (label >= 0){

                  for (int j = 0; j < temp->voxels_->points.size(); ++j){
                      temp->voxels_->points.at(j).r = clusterer.colors[label][0];
                      temp->voxels_->points.at(j).g = clusterer.colors[label][1];
                      temp->voxels_->points.at(j).b = clusterer.colors[label][2];
                      viscloud->push_back(temp->voxels_->points.at(j));
                    }
                }
              else{
                  for (int j = 0; j < temp->voxels_->points.size(); ++j){
                      temp->voxels_->points.at(j).r = 255;
                      temp->voxels_->points.at(j).g = 255;
                      temp->voxels_->points.at(j).b = 255;
                      viscloud->push_back(temp->voxels_->points.at(j));
                    }
                }
            }
          processed+=ids[l][s].size();
          viscloud->width = 1;
          viscloud->height = viscloud->points.size();
          pcl::io::savePCDFileASCII ("/media/chris/Seagate Backup Plus Drive/results/clustering/" + scans[s].stem().string() +std::to_string(l) +  "_clusters.pcd", *viscloud);
        }


      auto dsCalcStart = std::chrono::high_resolution_clock::now();
      processed = 0;
      // Calculate distribution and similarity scores for each scan
      for (int s = 0; s < scans.size(); ++s){

          // Distribution
          std::vector<float> clusterCount(clusterer.n_clusters);
          std::vector<float> clusterDistr(clusterer.n_clusters);

          int totalSv = ids[l][s].size();
          int totalCount = 0;
          for (int k = 0; k < ids[l][s].size() ; ++k){
              int label = clusterer.labels[k+processed];
              clusterCount[label]++;
              totalCount++;
            }

          assert (totalCount == totalSv);

          for (int c = 0; c < clusterCount.size() ; ++c){
              clusterDistr[c] = clusterCount[c] / totalSv;
              clusterDistr[c] *= 100; // convert to percentage
            }


          // Cosine similarity
          std::vector<float> clusterSim(clusterer.n_clusters);

          for (int k = 0; k < ids[l][s].size() ; ++k){
              int label = clusterer.labels[k+processed];
              cv::Mat center = clusterer.centers.row(label);

              double similarity = cosSimilarity(samples[l].row(k+processed),center);
              assert (similarity >= -1 && similarity <= 1);
              clusterSim[label]+=similarity;
            }

          for (int k = 0; k < clusterer.n_clusters; ++k){
              clusterSim[k] /= clusterCount[k];
              assert (clusterSim[k] >= -1 && clusterSim[k] <= 1);
            }

          allDistr.push_back(clusterDistr);
          allSim.push_back(clusterSim);
          processed+=ids[l][s].size();
        }

      auto dsCalcEnd = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> dsCalcTime = dsCalcEnd - dsCalcStart;
      std::cout << "Distance & similarity calc took " << dsCalcTime.count() << "s" << std::endl;

    }

  // Print  cluster distributions and similarities
  for (int i = 0; i < allDistr.size(); ++i){
      std::cout << scans[i].stem().string() << ": ";
      for (int j = 0; j < allDistr[i].size(); ++j){
          std::cout << allDistr[i][j] << " ";
        }
      for (int j = 0; j < allSim[i].size(); ++j){
          std::cout << allSim[i][j] << " ";
        }
      std::cout << std::endl;
    }


  auto selectionStart = std::chrono::high_resolution_clock::now();

  // "Balanced" scheme
  if (selectionScheme == 1){

      std::vector<float> balanceScores;

      for (int i = 0; i < allDistr.size(); ++i){
          float balance = 0;

          for (int j = 0; j < allDistr[i].size(); ++j){
              balance += abs(allDistr[i][j] - balanceGoal);
            }
          std::cout << "Balance score for " << scans[i].stem().string() << ": " << balance << std::endl;
          balanceScores.push_back(balance);
        }

      std::vector<int> indices(balanceScores.size());
      std::size_t num(0);
      std::generate(std::begin(indices), std::end(indices), [&]{ return num++; });
      std::sort(std::begin(indices),  std::end(indices),[&](float i1, float i2) { return balanceScores[i1] < balanceScores[i2]; } );

      std::cout << "Selected scans:" << std::endl;
      for (int i = 0 ; i < n; ++i){
          selected.push_back(scans.at(indices[i]));
          std::cout << selected[i].stem() << ": " << balanceScores[indices[i]]  << std::endl;
        }
    }


  // "Similarity" scheme
  else if (selectionScheme == 2){

      std::vector<float> simScores;

      for (int i = 0; i < allSim.size(); ++i){
          float simScore = 0;

          for (int j = 0; j < allSim[i].size(); ++j){
              simScore += allSim[i][j];
            }
          std::cout << "Simalarity score for " << scans[i].stem().string() << ": " << simScore << std::endl;
          simScores.push_back(simScore);
        }

      std::vector<int> indices(simScores.size());
      std::size_t num(0);
      std::generate(std::begin(indices), std::end(indices), [&]{ return num++; });

      std::sort(std::begin(indices),  std::end(indices),[&](float i1, float i2) { return simScores[i1] > simScores[i2]; } );

      std::cout << "Selected scans:" << std::endl;
      for (int i = 0 ; i < n; ++i){
          selected.push_back(scans.at(indices[i]));
          std::cout << selected[i].stem() << ": " << simScores[indices[i]]  << std::endl;
        }
    }


  // "Distinct" scheme
  else if (selectionScheme == 3){

      Clustering scancluster;

      cv::Mat scanFeat(allDistr.size(),allDistr[0].size() + allSim[0].size(),CV_32FC1);

      for (int i = 0; i < scanFeat.rows; ++i)
        for (int j = 0; j < allDistr[i].size(); ++j){
            scanFeat.at<float>(i,j) = allDistr[i][j];
            scanFeat.at<float>(i,j+allDistr[i].size()) = allSim[i][j];
          }

      // Standardize samples: x' = (x-mean)/std
      cv::Scalar mean,std;
      for (int i = 0; i < scanFeat.cols ; ++i){
          cv::meanStdDev(scanFeat.col(i), mean,std);
          for (int j = 0; j < scanFeat.col(i).rows; ++j)
            scanFeat.at<float>(j,i) = (scanFeat.at<float>(j,i)-mean[0])/std[0];
        }

      std::cout << "Performing KMeans on scan features..." << std::endl;
      scancluster.setSamples(scanFeat);
      int k = n;
      int iterations = 0;
      std::vector<int> min_indices;
      scancluster.kmeansNoEmpty(k, iterations);

      std::cout << "Size of scan clusters: " << std::endl;
      for (int i = 0; i < scancluster.n_clusters; ++i)
        std::cout << "Cluster " << i << " size: " << scancluster.clusterSizes[i] << std::endl;

      // Find scans closest to each cluster
      for (int i = 0; i < scancluster.centers.rows; ++i) {
          cv::Mat center = scancluster.centers.row(i);
          float min = std::numeric_limits<float>::max();
          int minidx;

          for (int j = 0; j < scanFeat.rows; ++j) {
              cv::Mat sample = scanFeat.row(j);

              float dist = cv::norm(sample,center,cv::NORM_L2);

              if (dist < min){
                  min = dist;
                  minidx = j;
                }
            }
          min_indices.push_back(minidx);
          selected.push_back(scans.at(minidx));
        }

      // Print selected scans and their feature vectors
      std::cout << "Selected scans + their features before scaling:" << std::endl;
      for (int i = 0; i < selected.size(); ++i){
          std::cout << selected[i].stem() << ": ";
          for (int j = 0; j < allDistr[i].size(); ++j){
              std::cout << allDistr[min_indices[i]][j] << " ";
            }
          for (int j = 0; j < allSim[i].size(); ++j){
              std::cout << allSim[min_indices[i]][j] << " ";
            }
          std::cout << std::endl;
        }

      // Print selected scans and their feature vectors
      std::cout << "Selected scans + their features after scaling:" << std::endl;
      for (int i = 0; i < selected.size(); ++i){
          std::cout << selected[i].stem() << ": ";

          for (int j = 0; j < scanFeat.cols; ++j)
            std::cout << scanFeat.at<float>(min_indices[i],j) << " ";

          std::cout << std::endl;
        }
    }

  auto selectionEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> selectionTime = selectionEnd - selectionStart;
  std::cout << "Selection took " << selectionTime.count() << "s" << std::endl;

  return selected;

}
