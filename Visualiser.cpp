#include "Visualiser.h"

Visualiser::Visualiser (){
  colours.resize(10);
  colours[0] = {255,0,0}; // red
  colours[1] = {0,255,0}; // lime
  colours[2] = {0,0,255}; // blue
  colours[3] = {0,255,255}; // aqua
  colours[4] = {255,255,0}; // yellow
  colours[5] = {0,128,128}; // teal
  colours[6] = {128,128,128}; // grey
  colours[7] = {0,128,0}; // green
  colours[8] = {255,0,255}; // fuchsia
  colours[9] = {255,255,255}; // white
}

Visualiser::~Visualiser (){
}

void Visualiser::setTestCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr & cloud){
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr tempcloud (new pcl::PointCloud<pcl::PointXYZRGBL> ());
  pcl::copyPointCloud(*cloud, *tempcloud);
  viscloud = tempcloud;
}

void Visualiser:: labelPoint(int idx, int result){
  viscloud->at(idx).label = result+1; // +1 to differentiate class 0 from unlabeled points
  viscloud->at(idx).r = colours[result][0];
  viscloud->at(idx).g = colours[result][1];
  viscloud->at(idx).b = colours[result][2];
}

void Visualiser::visFeature(int idx, float scale){
  float r = 255*scale;
  float g = 255-r;
  float b = 0;
  int newRgb = ((int(r)&0x0ff)<<16)|((int(g)&0x0ff)<<8)|(int(b)&0x0ff);
  viscloud->at(idx).rgb = newRgb;
}

void Visualiser::savePredictionCloud(std::string path, std::string currentfile){
  const std::string file = path + "predictions/" + currentfile + "_prediction.pcd";
  viscloud->width = 1;
  viscloud->height = viscloud->points.size();





  pcl::io::savePCDFileASCII (file, *viscloud);
}

void Visualiser::saveErrorCloud(std::string path, std::string labelpath, std::string currentfile){
  const std::string ground_truth_labels = labelpath + currentfile + ".labels";
  const std::string file = path + "/predictions/" + currentfile + "_errors.pcd";
  const std::string truth_file = path + "/ground_truth/" + currentfile + "_ground_truth.pcd";
  const std::string current_truth_prediction = path + "/predictions/" +  currentfile + "_truth_pred.csv";
  const std::string all_truth_prediction = path + "/predictions/all_truth_pred.csv";

  std::ofstream ofile; // for entire datasets predictions
  std::ofstream ofile2; // for current predictions only

  ofile2.open(current_truth_prediction);


  std::ifstream ifile(ground_truth_labels);

  if (startedWriting)
    ofile.open(all_truth_prediction, std::ios_base::app);
  else
    {
      ofile.open(all_truth_prediction);
      startedWriting = true;
    }

  std::cout << "Loading ground truth labels from " << ground_truth_labels << std::endl;

  int label;
  float correct = 0;
  float totalLabelled = 0;
  int idx = 0;

  while (ifile >> label){

      if (label!=99 && viscloud->points.at(idx).label != 255) { // If label exists and cloud is labelled
          ofile << label+1 << ",";
          ofile2 << label+1 << ",";
          totalLabelled++;

          ofile << viscloud->points.at(idx).label << "\n";
          ofile2 << viscloud->points.at(idx).label << "\n";
        }


      if (label+1 == viscloud->points.at(idx).label){ // If prediction == label
          correct++;
        }
      else if (label == 99) // Else, if no ground truth exists paint the point white
        {
          viscloud->points.at(idx).label = 99;
          viscloud->at(idx).r = colours[9][0];
          viscloud->at(idx).g = colours[9][1];
          viscloud->at(idx).b = colours[9][2];
        }
      else{ // Else, paint point fuchsia for incorrect prediction
          viscloud->at(idx).r = colours[8][0];
          viscloud->at(idx).g = colours[8][1];
          viscloud->at(idx).b = colours[8][2];
        }

      idx++;
    }

  ifile.close();
  ofile.close();
  ofile2.close();

  viscloud->width = 1;
  viscloud->height = viscloud->points.size();
  pcl::io::savePCDFileASCII (file, *viscloud);

  std::cout << "Saved error cloud. " << (correct/totalLabelled)*100 << "% of (labelled) ground truth points correctly classified." << std::endl;

  ifile.open(ground_truth_labels);
  idx=0;
  while (ifile >> label){
      if (label != 99){ // If ground truth point is labelled paint it its class colour
          viscloud->points.at(idx).label = label;
          viscloud->at(idx).r = colours[label][0];
          viscloud->at(idx).g = colours[label][1];
          viscloud->at(idx).b = colours[label][2];
        }
      else{ // Else, paint the point white
          viscloud->points.at(idx).label = label;
          viscloud->at(idx).r = colours[9][0];
          viscloud->at(idx).g = colours[9][1];
          viscloud->at(idx).b = colours[9][2];
        }
      idx++;
    }

  ifile.close();

  viscloud->width = 1;
  viscloud->height = viscloud->points.size();
  pcl::io::savePCDFileASCII (truth_file, *viscloud);

  std::cout << "Saved ground truth cloud." << std::endl;

}

