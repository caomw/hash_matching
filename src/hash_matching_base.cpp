#include "hash_matching_base.h"
#include "stereo_properties.h"
#include "opencv_utils.h"
#include <boost/shared_ptr.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

namespace fs=boost::filesystem;

hash_matching::HashMatchingBase::HashMatchingBase(
  ros::NodeHandle nh, ros::NodeHandle nhp) : nh_(nh), nh_private_(nhp)
{
  // Load parameters
  string ref_path, img_dir, desc_type, files_path;
  double desc_thresh;
  int num_hyperplanes, bucket_width, bucket_height, bucket_max, best_n;
  nh_private_.param("ref_path", ref_path, std::string(""));
  nh_private_.param("img_dir", img_dir, std::string(""));
  nh_private_.param("desc_type", desc_type, std::string("SIFT"));
  nh_private_.getParam("desc_thresh", desc_thresh);
  nh_private_.getParam("best_n", best_n);
  nh_private_.getParam("num_hyperplanes", num_hyperplanes);
  nh_private_.getParam("bucket_width", bucket_width);
  nh_private_.getParam("bucket_height", bucket_height);
  nh_private_.getParam("bucket_max", bucket_max);
  nh_private_.param("files_path", files_path, std::string("/home/user"));

  // Files path sanity check
  if (files_path[files_path.length()-1] != '/')
    files_path += "/";

  // Define image properties
  StereoProperties ref_prop;
  StereoProperties cur_prop;

  // Setup the parameters
  hash_matching::StereoProperties::Params image_params;
  image_params.desc_type = desc_type;
  image_params.num_hyperplanes = num_hyperplanes;
  image_params.bucket_width = bucket_width;
  image_params.bucket_height = bucket_height;
  image_params.bucket_max = bucket_max;
  ref_prop.setParams(image_params);
  cur_prop.setParams(image_params);

  // Sanity checks
  if (!boost::filesystem::exists(ref_path) || ref_path == "")
  {
    ROS_ERROR_STREAM("[HashMatching:] The reference image file does not exists: " << 
                     ref_path);
    return;
  }
  if (!fs::exists(img_dir) || !fs::is_directory(img_dir)) 
  {
    ROS_ERROR_STREAM("[HashMatching:] The image directory does not exists: " << 
                     img_dir);
    return;
  }

  // Initialize the output
  ostringstream output_csv;

  // Read the template image and extract kp and descriptors
  Mat img_temp = imread(ref_path, CV_LOAD_IMAGE_COLOR);
  ref_prop.setImage(img_temp);
  ref_prop.computeHyperplanes();
  ref_prop.computeHash();
  ROS_INFO_STREAM("Reference Keypoints Size: " << ref_prop.getKp().size());
  
  // Loop directory images
  fs::directory_iterator it(img_dir);
  fs::directory_iterator end;
  vector< pair<double,string> > dists;
  while (it!=end)
  {
    // Check if the directory entry is an directory.
    if (!fs::is_directory(it->status())) 
    {
      // Read image
      string path = img_dir + "/" + it->path().filename().string();
      Mat img_cur = imread(path, CV_LOAD_IMAGE_COLOR);
      cur_prop.setImage(img_cur);
      cur_prop.setHyperplanes(ref_prop.getCentroid(), ref_prop.getH(), ref_prop.getDelta());

      // Crosscheck matchin
      vector<DMatch> matches;
      Mat match_mask;
      hash_matching::OpencvUtils::crossCheckThresholdMatching(ref_prop.getDesc(), 
                                                              cur_prop.getDesc(), 
                                                              desc_thresh, 
                                                              match_mask, matches);
      // Compute hash
      cur_prop.computeHash();

      // Compare hashes
      double hash_matching1 = match(ref_prop.getHash1(), cur_prop.getHash1());
      double hash_matching2 = match(ref_prop.getHash2(), cur_prop.getHash2());
      double hash_matching3 = match(ref_prop.getHash3(), cur_prop.getHash3());
      double matching = hash_matching1*hash_matching2*hash_matching3;

      if (isfinite(matching))
      {
        // Log
        ROS_INFO_STREAM(it->path().filename().string() << " -> Hash: " <<
          matching << " | Desc. Matches: " << (int)matches.size());

        output_csv << matching << "," << matches.size() << endl; 

        // Save hash into vector
        dists.push_back(make_pair(matching, it->path().filename().string()));
      }
    }
    // Next directory entry
    it++;
  }

  // Save data into file
  string out_file;
  out_file = files_path + "output.txt";
  fstream f_out(out_file.c_str(), ios::out | ios::trunc);
  f_out << output_csv.str();
  f_out.close();

  // Sort distances vector
  sort(dists.begin(), dists.end(), hash_matching::OpencvUtils::sortByDistance);

  // Show result
  for(int i=0; i<best_n; i++)
    ROS_INFO_STREAM("BEST MATCHING: " << dists[i].second << " (" << dists[i].first << ")");

  ROS_INFO("FINISH!");
}

double hash_matching::HashMatchingBase::match(vector<uint> hash_1, vector<uint> hash_2)
{
  // Sanity check
  ROS_ASSERT(hash_1.size() == hash_2.size());

  double sum = 0.0;
  for (uint i=0; i<hash_1.size(); i++)
  {
    sum += pow((double)hash_1[i] - (double)hash_2[i], 2.0);
  }
  return sqrt(sum);
}
double hash_matching::HashMatchingBase::match(vector<double> hash_1, vector<double> hash_2)
{
  // Sanity check
  ROS_ASSERT(hash_1.size() == hash_2.size());

  double sum = 0.0;
  for (uint i=0; i<hash_1.size(); i++)
  {
    sum += pow((double)hash_1[i] - (double)hash_2[i], 2.0);
  }
  return sqrt(sum);
}

