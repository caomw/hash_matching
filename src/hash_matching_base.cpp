#include "hash_matching_base.h"
#include "stereo_properties.h"
#include "opencv_utils.h"
#include "hash.h"
#include <boost/shared_ptr.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/filesystem.hpp>

namespace fs=boost::filesystem;

hash_matching::HashMatchingBase::HashMatchingBase(
  ros::NodeHandle nh, ros::NodeHandle nhp) : nh_(nh), nh_private_(nhp)
{
  // Load parameters
  string ref_path, img_dir, desc_type, files_path;
  double desc_thresh;
  int best_n, proj_num, features_max_value, n_levels;
  nh_private_.param("files_path", files_path, std::string("/home/user"));
  nh_private_.param("ref_path", ref_path, std::string(""));
  nh_private_.param("img_dir", img_dir, std::string(""));
  nh_private_.param("desc_type", desc_type, std::string("SIFT"));
  nh_private_.getParam("desc_thresh", desc_thresh);
  nh_private_.getParam("best_n", best_n);
  nh_private_.getParam("proj_num", proj_num);
  nh_private_.getParam("features_max_value", features_max_value);
  nh_private_.getParam("n_levels", n_levels);

  // Files path sanity check
  if (files_path[files_path.length()-1] != '/')
    files_path += "/";

  // Sanity checks
  if (!fs::exists(ref_path) || ref_path == "")
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

  // Get the reference image filename
  fs::path ref_img_path(ref_path);
  string ref_img_fn = ref_img_path.filename().string();
  string ref_image_name = ref_img_fn.substr(0, ref_img_fn.find_last_of("."));

  // Initialize image properties
  StereoProperties ref_prop, cur_prop;
  hash_matching::StereoProperties::Params image_params;
  image_params.desc_type = desc_type;
  ref_prop.setParams(image_params);
  cur_prop.setParams(image_params);

  // Initialize hash
  Hash hash_obj;
  hash_matching::Hash::Params hash_params;
  hash_params.proj_num = proj_num;
  hash_params.features_max_value = features_max_value;
  hash_params.n_levels = n_levels;
  hash_obj.setParams(hash_params);

  // Read the template image and extract kp and descriptors
  Mat img_temp = imread(ref_path, CV_LOAD_IMAGE_COLOR);
  ref_prop.setImage(img_temp);

  // Compute the reference hash
  hash_obj.initialize(ref_prop.getDesc());
  vector<uint> ref_hash_1 = hash_obj.getHash1(ref_prop.getDesc());
  vector<uint> ref_hash_2 = hash_obj.getHash2(ref_prop.getDesc());
  vector<float> ref_hash_3 = hash_obj.getHash3(ref_prop.getDesc());

  // Initialize the output
  ostringstream output_csv;

  // Initialize hash distances
  vector<trio> dists_1, dists_2, dists_3;

  // Sort directory of images
  typedef std::vector<fs::path> vec;
  vec v;
  copy(
        fs::directory_iterator(img_dir), 
        fs::directory_iterator(),
        back_inserter(v)
      );

  sort(v.begin(), v.end());
  vec::const_iterator it(v.begin());
  
  // Iterate over all images
  while (it!=v.end())
  {
    // Check if the directory entry is an directory.
    if (!fs::is_directory(*it)) 
    {
      // Get filename
      string filename = it->filename().string();
      int lastindex = filename.find_last_of(".");
      string rawname = filename.substr(0, lastindex);

      // Read image
      string path = img_dir + "/" + filename;
      Mat img_cur = imread(path, CV_LOAD_IMAGE_COLOR);
      cur_prop.setImage(img_cur);

      // Hash 1
      ros::WallTime st_hash_1 = ros::WallTime::now();
      vector<uint> cur_hash_1 = hash_obj.getHash1(cur_prop.getDesc());
      ros::WallDuration time_hash_1 = ros::WallTime::now() - st_hash_1;

      // Hash 2
      ros::WallTime st_hash_2 = ros::WallTime::now();
      vector<uint> cur_hash_2 = hash_obj.getHash2(cur_prop.getDesc());
      ros::WallDuration time_hash_2 = ros::WallTime::now() - st_hash_2;

      // Hash 3
      ros::WallTime st_hash_3 = ros::WallTime::now();
      vector<float> cur_hash_3 = hash_obj.getHash3(cur_prop.getDesc());
      ros::WallDuration time_hash_3 = ros::WallTime::now() - st_hash_3;

      // Crosscheck matching
      vector<DMatch> matches;
      Mat match_mask;
      ros::WallTime start_time_desc = ros::WallTime::now();
      hash_matching::OpencvUtils::crossCheckThresholdMatching(ref_prop.getDesc(), 
                                                              cur_prop.getDesc(), 
                                                              desc_thresh, 
                                                              match_mask, matches);
      ros::WallDuration time_desc_matching = ros::WallTime::now() - start_time_desc;
  
      // Compare hashes
      st_hash_1 = ros::WallTime::now();
      uint matching_1 = match(ref_hash_1, cur_hash_1);
      time_hash_1 = time_hash_1 + (ros::WallTime::now() - st_hash_1);

      st_hash_2 = ros::WallTime::now();
      uint matching_2 = match(ref_hash_2, cur_hash_2);
      time_hash_2 = time_hash_2 + (ros::WallTime::now() - st_hash_2);

      st_hash_3 = ros::WallTime::now();
      float matching_3 = match(ref_hash_3, cur_hash_3);
      time_hash_3 = time_hash_3 + (ros::WallTime::now() - st_hash_3);

      // Log
      ROS_INFO_STREAM(rawname << " -> Hash: " <<
        matching_1 << " | " << matching_2 << " | " << matching_3 << " | Desc. Matches: " << (int)matches.size());
      output_csv << ref_image_name << "," << rawname << "," << matches.size() <<
                    "," << matching_1 << "," << matching_2 << "," << matching_3 <<
                    "," << time_hash_1.toSec() << "," << time_hash_2.toSec() <<
                    "," << time_hash_3.toSec() << "," << time_desc_matching.toSec() << endl;

      // Save hash into vector
      trio comp_1((float)matching_1, (int)matches.size(), rawname);
      trio comp_2((float)matching_2, (int)matches.size(), rawname);
      trio comp_3((float)matching_3, (int)matches.size(), rawname);
      dists_1.push_back(comp_1);
      dists_2.push_back(comp_2);
      dists_3.push_back(comp_3);
    }
    // Next directory entry
    it++;
  }

  // Save data into file
  string out_file;
  out_file = files_path + desc_type + "_" + toString(hash_obj.getHyperplanes()) + "_" + toString(proj_num) + "_" + ref_image_name + ".txt";
  fstream f_out(out_file.c_str(), ios::out | ios::trunc);
  f_out << output_csv.str();
  f_out.close();

  // Sort distances vector
  sort(dists_1.begin(), dists_1.end(), hash_matching::OpencvUtils::sortTrioByDistance);
  sort(dists_2.begin(), dists_2.end(), hash_matching::OpencvUtils::sortTrioByDistance);
  sort(dists_3.begin(), dists_3.end(), hash_matching::OpencvUtils::sortTrioByDistance);

  // Show result
  ROS_INFO("###################################################");
  for(int i=0; i<best_n; i++)
    ROS_INFO_STREAM("BEST MATCHING HASH 1: " << dists_1[i].image << ": " << dists_1[i].hash_matching << " (" << dists_1[i].feature_matchings << ")");
  ROS_INFO("-------------------");
  for(int i=0; i<best_n; i++)
    ROS_INFO_STREAM("BEST MATCHING HASH 2: " << dists_2[i].image << ": " << dists_2[i].hash_matching << " (" << dists_2[i].feature_matchings << ")");
  ROS_INFO("-------------------");
  for(int i=0; i<best_n; i++)
    ROS_INFO_STREAM("BEST MATCHING HASH 3: " << dists_3[i].image << ": " << dists_3[i].hash_matching << " (" << dists_3[i].feature_matchings << ")");
  ROS_INFO("###################################################");

  // Close the node
  ros::shutdown();
}

// Hash matching functions
float hash_matching::HashMatchingBase::match(vector<float> hash_1, vector<float> hash_2)
{
  ROS_ASSERT(hash_1.size() == hash_2.size()); // Sanity check

  // Compute the distance
  float sum = 0.0;
  for (uint i=0; i<hash_1.size(); i++)
    sum += fabs(hash_1[i] - hash_2[i]);

  return sum;
}
uint hash_matching::HashMatchingBase::match(vector<uint> hash_1, vector<uint> hash_2)
{
  ROS_ASSERT(hash_1.size() == hash_2.size()); // Sanity check

  // Compute the distance
  uint sum = 0;
  for (uint i=0; i<hash_1.size(); i++)
    sum += abs(hash_1[i] - hash_2[i]);

  return sum;
}

// Converts number to string
template <typename T> string hash_matching::HashMatchingBase::toString( const T& n )
{
  ostringstream stm ;
  stm << n ;
  return stm.str() ;
}