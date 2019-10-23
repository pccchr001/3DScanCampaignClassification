#include "CloudReader.h"
#include <cstdio>
#include <fstream>

CloudReader::CloudReader(){
}

CloudReader::~CloudReader() {
}

// Read PTX to XYZRGBA PointCloud
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr CloudReader::readToCloud(boost::filesystem::path & file_name)
{
  std::fstream ifile(file_name.c_str(), std::fstream::in);
  std::ofstream ofile;

  if (file_name.extension().string() == ".xyz_label_conf")
    ofile.open("/media/chris/Seagate Backup Plus Drive/labels/" + file_name.stem().string() + ".labels");

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

  std::map<int,int> labelmap;
  labelmap[1200] = 1;
  labelmap[1400] = 2;
  labelmap[1004] = 3;
  labelmap[1103] = 4;
  labelmap[1100] = 5;

  int skip = 0;
  if (file_name.extension().string() == ".ptx")
    skip = 10;

  std::string line;
  for(int i = 0; i < skip; i++) // Skip lines
    getline(ifile,line);

  pcl::PointXYZRGBA p;
  float intens, label;
  int r,g,b;
  while (getline(ifile,line)){
      std::stringstream ss(line);
      ss >> p.x >> p.y >> p.z;
      if (file_name.extension().string() == ".ptx"){
          ss >> intens >> r >> g >> b;
          p.rgba = (r << 16 | g << 8 | b);
        }
      else if (file_name.extension().string() == ".xyz_label_conf"){
          ss >> label;
          assert (label == 1200 || label ==1400 || label ==1004 || label ==1103|| label ==1100);


          p.rgba = (255 << 16 | 255 << 8 | 255);
          ofile << labelmap[label] << "\n";
        }

      cloud->points.push_back(p);
    }
  ifile.close();
  if (ofile.is_open())ofile.close();
  return cloud;
}

// Read PTX to XYZRGBA PointCloud
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr CloudReader::readSemanticToCloud(boost::filesystem::path & file_name)
{
  std::fstream ifile(file_name.c_str(), std::fstream::in);

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

  int skip = 0;
  if (file_name.extension().string() == ".txt")
    skip = 0;

  std::string line;
  for(int i = 0; i < skip; i++) // Skip lines
    getline(ifile,line);

  pcl::PointXYZRGBA p;
  while (getline(ifile,line)){
      std::stringstream ss(line);
      ss >> p.x >> p.y >> p.z;
      cloud->points.push_back(p);
    }
  ifile.close();

  return cloud;
}

// Read Keep/Discard PTX to XYZRGBA PointCloud
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr CloudReader::readKDToCloud(boost::filesystem::path & file_name, std::string labelsOut)
{
  std::fstream ifile(file_name.c_str(), std::fstream::in);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

  std::ofstream ofile(labelsOut);

  int skip = 0;
  if (file_name.extension().string() == ".ptx")
    skip = 10;

  std::string line;
  for(int i = 0; i < skip; i++) // Skip lines
    getline(ifile,line);

  pcl::PointXYZRGBA p;
  float intens;
  int r,g,b;
  int label;

  while (getline(ifile,line)){
      std::stringstream ss(line);
      ss >> p.x >> p.y >> p.z >> intens;
      ss >> label; // In the KD PTX scans, the R value is the keep/discard label
      r = g = b = 255;
      p.rgba = (r << 16 | g << 8 | b);
      cloud->points.push_back(p);

      ofile << label+1 << "\n"; // add 1 to label (label's are 0 or 1)

    }
  ifile.close();
  ofile.close();
  return cloud;
}


// Split cloud into quadrants using x and y coordinates
void CloudReader::splitCloud(boost::filesystem::path labelPath, boost::filesystem::path  file_name)
{

  std::fstream labelFile(labelPath.string() + file_name.stem().string() + ".labels", std::fstream::in);
  std::fstream cloudFile(file_name.c_str(), std::fstream::in);

  std::ofstream labels1(labelPath.string() + file_name.stem().string() + "_1.labels");
  std::ofstream labels2(labelPath.string() +  file_name.stem().string() + "_2.labels");
  std::ofstream labels3(labelPath.string() + file_name.stem().string() + "_3.labels");
  std::ofstream labels4(labelPath.string() +  file_name.stem().string() + "_4.labels");

  //std::ofstream labels1(labelPath.string() + file_name.stem().string() + "_a.labels");
  //std::ofstream labels2(labelPath.string() +  file_name.stem().string() + "_b.labels");

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud3(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud4(new pcl::PointCloud<pcl::PointXYZRGBA>);

  int skip = 11;
  std::string line;
  for(int i = 0; i < skip; i++) // Skip lines
    getline(cloudFile,line);

  int label;
  pcl::PointXYZRGBA p;

  while (getline(cloudFile,line)){

      std::stringstream ss(line);
      ss >> p.x >> p.y >> p.z;
      labelFile >> label;

      if (p.x > 0 && p.y > 0){
          cloud1->points.push_back(p);
          labels1 << label << std::endl;
        }
      else if (p.x <= 0 && p.y > 0){
          cloud2->points.push_back(p);
          labels2 << label << std::endl;
        }
      else if (p.y <= 0 && p.x <= 0){
          cloud3->points.push_back(p);
          labels3 << label << std::endl;
        }
      else if (p.x > 0 && p.y <= 0){
          cloud4->points.push_back(p);
          labels4 << label << std::endl;
        }
    }

  cloudFile.close();
  labelFile.close();

  labels1.close();
  labels2.close();
  labels3.close();
  labels4.close();

  cloud1->width = 1;
  cloud1->height = cloud1->points.size();
  pcl::io::savePCDFileASCII (file_name.parent_path().string() + "/" + file_name.stem().string() + "_1.pcd", *cloud1);

  cloud2->width = 1;
  cloud2->height = cloud2->points.size();
  pcl::io::savePCDFileASCII (file_name.parent_path().string() + "/" + file_name.stem().string() + "_2.pcd", *cloud2);

  cloud3->width = 1;
  cloud3->height = cloud3->points.size();
  pcl::io::savePCDFileASCII (file_name.parent_path().string() + "/" + file_name.stem().string() + "_3.pcd", *cloud3);

  cloud4->width = 1;
  cloud4->height = cloud4->points.size();
  pcl::io::savePCDFileASCII (file_name.parent_path().string() + "/" + file_name.stem().string() + "_4.pcd", *cloud4);

}


// Read PTX to RGB Mat
cv::Mat CloudReader::readToRGB(const std::string & file_name){
  std::fstream ifile(file_name.c_str(), std::fstream::in);
  int height,width;
  ifile >> height;
  ifile >> width;

  cv::Mat rgb_image(width,height, CV_8UC3);

  std::string line;
  std::string junk;
  int r,g,b;

  for(int i = 0; i < 8; i++ )
    getline(ifile,line);

  for (int i = 0; i < height; ++i){
      for (int j = 0; j < width; ++j){
          getline(ifile,line);
          std::stringstream ss(line);

          for(int x = 0; x < 4; x++ ) // Ignore XYZI
            ss >> junk;

          ss >> r >> g >> b;

          rgb_image.at<cv::Vec3b>(j, i).val[0] = b;
          rgb_image.at<cv::Vec3b>(j, i).val[1] = g;
          rgb_image.at<cv::Vec3b>(j, i).val[2] = r;
        }
    }
  ifile.close();
  return rgb_image;
}
