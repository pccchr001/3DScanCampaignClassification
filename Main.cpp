// Source headers
#include "Common.h"
#include "CloudReader.h"
#include "Downsampler.h"
#include "KSearcher.h"
#include "CloudFex.h"
#include "TrainingLabels.h"
#include "TrainingData.h"
#include "ScanSelection.h"
#include "Visualiser.h"

// Classifiers
#include "classifiers/RandomForest.h"
#include "classifiers/SVM.h"
#include "classifiers/NeuralNet.h"

#include <chrono>
#include <thread>
#include <omp.h>

using namespace std;
using namespace pcl;

int main (int argc, char** argv)
{
  // Global values
  vector<boost::filesystem::path> raw_scans;
  vector<boost::filesystem::path> pcd_scans;

  boost::filesystem::path scanpath = "/media/chris/Seagate Backup Plus Drive/scans/";
  const boost::filesystem::path labelpath = "/media/chris/Seagate Backup Plus Drive/labels/";
  const boost::filesystem::path resultpath = "/media/chris/Seagate Backup Plus Drive/results/";

  std::cout << "train(t) or predict(p)?" << std::endl;
  char c; bool training;
  std::cin >> c;
  assert (c == 't' || c == 'p');
  if (c == 't'){
      training = true;
      scanpath += "training/";
    }
  else {
      training = false;
      scanpath += "testing/";
    }

  // Classifier and feature selection params
  int n_classes = 2;
  const int n_examples = 3000;
  NeuralNet classifier(n_classes);
  if (!training)classifier.loadModel("../data/models/classifier.yml");


  bool loadFeat = false;
  if (training){
      std::cout << "load previously stored features? (y or n)" << std::endl;
      char c;
      std::cin >> c;
      assert (c == 'y' || c == 'n');
      if (c == 'y') loadFeat = true;
    }

  // Scan selection params
  const bool scan_selection = true;
  int n_scan_select;

  int selectionScheme;
  if (training && scan_selection && !loadFeat){
      std::cout << "scan selection mode? (0, 1, 2 or 3)" << std::endl;
      std::cin >> selectionScheme;
      assert (selectionScheme >= 0 && selectionScheme <= 3);

      std::cout << "number of scans to select? " << std::endl;
      std::cin >> n_scan_select;
    }


  std::cout << "feature selection mode? (0, 1 or 2)" << std::endl;
  int fs_mode;
  std::cin >> fs_mode;

  assert (fs_mode == 0 || fs_mode == 1 || fs_mode == 2);

  std::string classString = typeid(classifier).name();
  bool rf = classString == "12RandomForest";

  bool calcImportantFeat = false;
  if (rf && fs_mode == 2 && training)
    calcImportantFeat = true;


  // Other parameters
  const int max_batch_size = 100000;
  const int max_k = 10;
  const int feat_per_level = 24;
  const int n_levels = 3;
  const float downsample_res = 0.025f;
  const float downsample_mult = 2.0f;
  const bool multiscale_selection = false;
  const bool ground_truth = true;
  const bool convert_raw = false;
  const bool split_cloud = false; // for splitting large scans in half/quarters
  const bool parameter_tuning = false;

  std::chrono::duration<double> fexTime;

  // --- Converts raw scans to pcd ---
  if (convert_raw){

      // Get all PTX's from scan folder
      if (exists (scanpath))
        for (auto it : recursive_directory_range (scanpath))
          if (it.path ().extension ().string () == ".ptx" || it.path ().extension ().string () == ".xyz_label_conf"
              || it.path ().extension ().string () == ".txt" || it.path ().extension ().string () == ".pcd") //pcd temp!
            raw_scans.push_back (it);

      for (size_t i = 0; i < raw_scans.size (); ++i){

          // Read ptx to point cloud
          PointCloud<PointXYZRGBA>::Ptr cloud (new PointCloud<PointXYZRGBA>);
          CloudReader reader;
          cout << "Reading scan " << i + 1 << "/" << raw_scans.size () << " to point cloud..." << endl;

          // Read original PTX cloud (use readKDToCloud if reading pre-labelled keep/discard ptx)
          //cloud = reader.readKDToCloud(raw_scans.at(i), labelpath.string () + raw_scans.at (i).stem().string() + ".labels");
          //cloud = reader.readToCloud(raw_scans.at(i));
          cloud = reader.readSemanticToCloud(raw_scans.at(i));

          cout << "Downsampling to " << downsample_res << "m resolution..." << endl;
          Downsampler ds (cloud);
          PointCloud<PointXYZRGBA>::Ptr downsampled_cloud (new PointCloud<PointXYZRGBA> ());

          boost::filesystem::path labelsIn = labelpath.string () + raw_scans.at (i).stem().string() + ".labels";
          boost::filesystem::path labelsOut = labelpath.string() + raw_scans.at(i).stem().string() + "_ds.labels";

          downsampled_cloud = ds.downsampleWithLabels(downsample_res, n_classes, labelsIn, labelsOut);

          std::string file = scanpath.string() + raw_scans.at(i).stem().string () + "_ds.pcd";

          downsampled_cloud->width = 1;
          downsampled_cloud->height = downsampled_cloud->points.size();
          io::savePCDFileASCII (file, *downsampled_cloud);

          if (split_cloud){
              cout << "Splitting scan..." << endl;
              boost::filesystem::path filepath = file;
              reader.splitCloud(labelpath,boost::filesystem::path(filepath));
            }
        }

      std::cout << "Finished converting raw scans to pcd and stored labels separately." << std::endl;

      exit(0); // Done converting, exit here
    }

  // Get all PCD's from scan folder
  if (exists (scanpath))
    for (auto it : recursive_directory_range (scanpath))
      if (it.path ().extension ().string () == ".pcd")
        pcd_scans.push_back (it);

  TrainingLabels tlabels (n_classes, pcd_scans.size ());
  TrainingData tdata;
  std::vector<boost::filesystem::path> selectedScans;

  // Get selected features from text file
  std::vector<int> selectedFeat;

  if (fs_mode != 0 && !calcImportantFeat){
      std::string selectedFeatPath;

      if (fs_mode ==1){
          selectedFeatPath = resultpath.string() + "filteredFeat.txt";
        }
      else if (fs_mode ==2){
          selectedFeatPath = resultpath.string() + "importantFeat.txt";
        }

      ifstream ifile(selectedFeatPath);
      std::string token;

      while (ifile >> token){
          selectedFeat.push_back(stoi(token));
        }
      ifile.close();
      std::cout << "Loaded selected features file." << std::endl;
    }


  if (training && loadFeat){ // If using previously saved features
      cv::FileStorage featureStorage("/media/chris/Seagate Backup Plus Drive/results/features.yml", cv::FileStorage::READ);

      std::vector<float> storedMeans, storedStds;
      std::vector<std::vector<float>> storedFeats;
      featureStorage["means"] >> storedMeans;
      featureStorage["stds"] >> storedStds;
      featureStorage["features"] >> storedFeats;
      featureStorage.release();

      tdata.setMeans(storedMeans);
      tdata.setStds(storedStds);

      if (fs_mode != 0 && !calcImportantFeat){ // trim feature vectors

          for (int i = 0; i < storedFeats.size(); ++i)
            {
              std::vector<float> tempFeats;
              tempFeats.reserve(selectedFeat.size()+1); // +1 for label

              for (int s = 0; s < selectedFeat.size(); ++s){
                  if (selectedFeat[s] == 1){
                      tempFeats.push_back(storedFeats[i][s]);
                    }
                }
              tempFeats.push_back(storedFeats[i].back()); // append label
              tdata.addTrainingData(tempFeats);
            }

        }
      else{
          tdata.setTrainingData(storedFeats);
        }
    }

  else{ // Else if not using previously saved features


      if (!scan_selection || !training)
        selectedScans = pcd_scans;

      // -- Scan selection, labelling & training sample balancing ---
      if (training){

          cout << "Found " << pcd_scans.size() << " PCD clouds" << endl;

          // Select scans to label or train from
          if (scan_selection){
              ScanSelection selector;
              selectedScans = selector.supervoxelSelect(n_scan_select, pcd_scans, selectionScheme, multiscale_selection);
            }

          // Add selected scans' labels to tlabels
          for (int i = 0; i < selectedScans.size(); ++i){
              ifstream ifile(labelpath.string() + selectedScans.at(i).stem().string () + ".labels");
              int label;
              int idx = 0;
              while (ifile >> label){
                  if (label >=0 && label <= n_classes)
                    tlabels.addLabel(label, i, idx);
                  idx++;
                }
              ifile.close();
            }

          cout << "Balancing training data..." << endl;
          tlabels.overSample (n_examples);
          tlabels.underSample (n_examples);
          tlabels.printTotals();
        }
      else
        {
          cv::FileStorage featureStorage("/media/chris/Seagate Backup Plus Drive/results/features.yml", cv::FileStorage::READ);
          std::vector<float> featMeans, featStds, tempMeans, tempStds;
          featureStorage["means"] >> featMeans;
          featureStorage["stds"] >> featStds;

          if (fs_mode != 0){ // If feature selection, trim mean and std vectors
              for (int s = 0; s < selectedFeat.size(); ++s){
                  if (selectedFeat[s] == 1){
                      tempMeans.push_back(featMeans[s]);
                      tempStds.push_back(featStds[s]);
                    }
                }
              classifier.setMeanStd(tempMeans, tempStds); // Load means and stds into classifier
            }
          else
            classifier.setMeanStd(featMeans, featStds); // Load means and stds into classifier
        }

      Visualiser vis;

      // --- Multiscale feature extraction and storing of features
      for (size_t i = 0; i < selectedScans.size(); ++i)
        {
          PointCloud<PointXYZRGBA>::Ptr cloud (new PointCloud<PointXYZRGBA> ());
          PointCloud<PointXYZRGBA>::Ptr base_cloud;
          KSearcher neighbour_finder[n_levels];
          Eigen::MatrixXd point_matrices[n_levels];
          CloudFex fex;
          PCDReader reader;

          cout << "Reading in " << selectedScans[i].string() << "..." << endl;
          reader.read (selectedScans[i].string(), *cloud);

          base_cloud = cloud; // Store base cloud


          if (!training){
              vis.setTestCloud(base_cloud);

              for (size_t s = 0; s < base_cloud->points.size (); s++)
                if (!isnan (base_cloud->at (s).x) && !isnan (base_cloud->at (s).y) && !isnan (base_cloud->at (s).z))
                  if (! (base_cloud->at (s).x == 0 && base_cloud->at (s).y == 0 && base_cloud->at (s).y == 0))
                    tlabels.addLabel(0, i, s);
            }

          // For each pointcloud pyramid level
          for (int d = 0; d < n_levels; ++d){

              if (d > 0){ // If an intermediate pyramid level
                  Downsampler ds (cloud);
                  float res = downsample_res * (pow (downsample_mult, d));
                  cout << "Downsampling scan "<< i + 1 << "/" << selectedScans.size () << " to " << res << "m resolution..." << endl;
                  PointCloud<PointXYZRGBA>::Ptr downsampled_cloud (new PointCloud<PointXYZRGBA> ());
                  downsampled_cloud = ds.downsample(res);
                  cloud = downsampled_cloud; // Overwrite cloud for next pyramid level
                }
              neighbour_finder[d].setInput (cloud);
              neighbour_finder[d].setProjectedInput(cloud);

              // Store downsampled cloud's points in matrix for this level
              point_matrices[d].resize (cloud->points.size (), 3);
              for (size_t s = 0; s < cloud->points.size (); s++){
                  if (!isnan (cloud->at (s).x) && !isnan (cloud->at (s).y) && !isnan (cloud->at (s).z)){
                      point_matrices[d] (s, 0) = cloud->at (s).x;
                      point_matrices[d] (s, 1) = cloud->at (s).y;
                      point_matrices[d] (s, 2) = cloud->at (s).z;
                    }
                }
            }

          fex.setPointMatrix (point_matrices, n_levels);

          // --- Feature Extraction ---
          for (int j = 0; j < n_classes; ++j){

              vector<int> examples = tlabels.getExamples (j, i); // Get examples for this scan and class

              if (examples.size () == 0){
                  cout << "Skipping feature extraction (class " << j + 1 << " not in this scan)" << endl;
                }

              else{
                  cout << "Finding neighbours and extracting features for class " << j + 1 << "..." << endl;

                  vector<vector<float>> vector_batch;
                  vector<int> batch_indices;

                  // For each example
                  for (int x = 0; x < examples.size (); ++x) {

                      if (!training && x%1000 == 0) // Give progress while extracting features for prediction
                        cout << x << "/" << examples.size() << endl;

                      PointXYZRGBA example = base_cloud->at (examples.at(x));
                      std::vector<float> featureVector(feat_per_level * n_levels + 1);

                      auto startFex = std::chrono::high_resolution_clock::now(); // start timing start of batch
                      #pragma omp parallel num_threads(std::thread::hardware_concurrency())
                      {
                        #pragma omp for //ordered schedule(dynamic)
                        for (int level = 0; level < n_levels; ++level){
                            vector<int> knn = neighbour_finder[level].KSearch (example, max_k-1);
                            float cylinder_r = 0.05f;
                            // get cylinders min and max h (rather than all neighbours)
                            vector<float> cylinder_h = neighbour_finder[level].CylinderSearch(example, cylinder_r);
                            fex.calcPointFeat (example, knn, cylinder_h, j, level, &featureVector);
                          }
                      }
                      auto finishFex = std::chrono::high_resolution_clock::now();
                      fexTime += finishFex - startFex;

                      // Trim feature vector if needed
                      if (fs_mode != 0 && !calcImportantFeat){ // if feature selection active but not calcing important feat
                          std::vector<float> temp;

                          temp.reserve(selectedFeat.size()+1); // +1 for label

                          for (int s = 0; s < selectedFeat.size(); ++s){
                              if (selectedFeat[s] == 1)
                                temp.push_back(featureVector[s]);
                            }
                          temp.push_back(featureVector.back()); // append label
                          featureVector = temp;
                        }

                      if (training){
                          // Store training feature vector
                          tdata.addTrainingData(featureVector);
                        }

                      // Prediction
                      else{

                          vector_batch.push_back(featureVector);
                          batch_indices.push_back(examples.at(x));

                          // If batch is big enough or this is the last point
                          if (vector_batch.size() == max_batch_size || x == examples.size() -1){

                              vector<int> results = classifier.predict(vector_batch, false);

                              for (u_int l = 0; l < results.size(); ++l){
                                  assert (results[l] < n_classes);
                                  vis.labelPoint(batch_indices[l], results[l]); // Label the batch
                                }
                              vector_batch.clear();
                              batch_indices.clear();

                            }
                        }
                    }
                }
            }

          if (!training){

              vis.savePredictionCloud(resultpath.string(), selectedScans.at(i).stem().string()) ;
              std::cout << "Finished classifying scan. Results cloud saved to " << resultpath.string() << std::endl;

              if (ground_truth){
                  std::cout << "Ground truth found, saving error cloud... " << std::endl;
                  //vis.saveErrorCloud(resultpath.string(), labelpath.string() + selectedScans.at(i).stem().string () + ".labels");
                  vis.saveErrorCloud(resultpath.string(), labelpath.string(), selectedScans.at(i).stem().string ());
                }
            }
        }
    }

  // Training
  if (training){

      cout << "Training classifier..." << endl;

      std::ofstream timingFile;
      timingFile.open(resultpath.string() + "timing.csv");
      timingFile << "training_fex," << fexTime.count() << std::endl;

      if (!loadFeat) tdata.normalizeFeat(); // Normalize features if they weren't loaded from disk
      classifier.loadTrainingData(tdata.getTrainingData()); // Load training data into classifier
      classifier.train(parameter_tuning, (fs_mode == 2)); // Start training classifier

      timingFile << "training_time," << classifier.getTraintime().count() << std::endl;
      timingFile.close();

      // Save for future classifiers
      if (!loadFeat){
          tdata.writeTrainingData("/media/chris/Seagate Backup Plus Drive/results/features.csv", false); // For feature selection script
          cv::FileStorage featureStorage("/media/chris/Seagate Backup Plus Drive/results/features.yml", cv::FileStorage::WRITE);
          featureStorage << "features" << tdata.getTrainingData();
          featureStorage << "means" << tdata.getMeans();
          featureStorage << "stds" << tdata.getStds();
          featureStorage.release();

          std::cout << "Saved features, means and stds to disk." << std::endl;
        }

    }
  else {
      std::cout << "Feature extraction took " << fexTime.count() << "s" << std::endl;
      std::ofstream timingFile;
      timingFile.open(resultpath.string() + "timing.csv", std::ios_base::app);
      timingFile << "prediction_fex," << fexTime.count() << std::endl;

      std::cout << "Prediction took " << classifier.getPredicttime().count() << "s" << std::endl;
      timingFile << "prediction_time," << classifier.getPredicttime().count() << std::endl;
      timingFile.close();
    }
}
