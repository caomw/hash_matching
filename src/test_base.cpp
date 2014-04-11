#include <ros/package.h>
#include <boost/shared_ptr.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "test_base.h"
#include "hash.h"

namespace fs=boost::filesystem;

hash_matching::TestBase::TestBase(
  ros::NodeHandle nh, ros::NodeHandle nhp) : nh_(nh), nh_private_(nhp)
{
  // Load parameters
  string img_dir, desc_type, files_path, gt_file;
  double desc_thresh;
  bool proj_orthogonal, validate, stereo_dataset;
  int proj_num, min_neighbour, n_levels, min_matches, min_inliers, gt_tolerance;
  nh_private_.param("files_path", files_path, std::string(""));
  nh_private_.param("img_dir", img_dir, std::string(""));
  nh_private_.param("gt_file", gt_file, std::string(""));
  nh_private_.param("desc_type", desc_type, std::string("SIFT"));
  nh_private_.getParam("desc_thresh", desc_thresh);
  nh_private_.getParam("proj_num", proj_num);
  nh_private_.param("proj_orthogonal", proj_orthogonal, true);
  nh_private_.getParam("min_neighbour", min_neighbour);
  nh_private_.getParam("n_levels", n_levels);
  nh_private_.getParam("min_matches", min_matches);
  nh_private_.getParam("min_inliers", min_inliers);
  nh_private_.param("validate", validate, false);
  nh_private_.param("stereo_dataset", stereo_dataset, false);
  nh_private_.getParam("gt_tolerance", gt_tolerance);

  // Log
  cout << "  files_path       = " << files_path << endl;
  cout << "  img_dir          = " << img_dir << endl;
  cout << "  desc_type        = " << desc_type << endl;
  cout << "  desc_thresh      = " << desc_thresh << endl;
  cout << "  proj_num         = " << proj_num << endl;
  cout << "  proj_orthogonal  = " << proj_orthogonal << endl;
  cout << "  min_neighbour    = " << min_neighbour << endl;
  cout << "  n_levels         = " << n_levels << endl;
  cout << "  min_matches      = " << min_matches << endl;
  cout << "  min_inliers      = " << min_inliers << endl;
  cout << "  validate         = " << validate << endl;
  cout << "  stereo_dataset   = " << stereo_dataset << endl;
  cout << "  gt_tolerance     = " << gt_tolerance << endl;

  // Files path sanity check
  if (files_path[files_path.length()-1] != '/')
    files_path += "/";

  // Sanity checks
  if (!fs::exists(img_dir) || !fs::is_directory(img_dir)) 
  {
    ROS_ERROR_STREAM("[HashMatching:] The image directory does not exists: " << 
                     img_dir);
  }

  // The feature min/max value depends on the type of descriptor
  float features_max_value, features_min_value;
  if (boost::iequals(desc_type, "sift") || boost::iequals(desc_type, "opponentsift"))
  {
    features_max_value = 255.0;
    features_min_value = 0.0;
  }
  else if (boost::iequals(desc_type, "surf") || boost::iequals(desc_type, "opponentsurf"))
  {
    features_max_value = 1.0;
    features_min_value = -1.0;
  }
  else if (boost::iequals(desc_type, "orb") || boost::iequals(desc_type, "opponentorb"))
  {
    features_max_value = 1.0;
    features_min_value = 0.0;
  }
  else if (boost::iequals(desc_type, "brisk") || boost::iequals(desc_type, "opponentbrisk"))
  {
    features_max_value = 1.0;
    features_min_value = 0.0;
  }
  else
  {
    ROS_ERROR("[HashMatching:] Descriptor type must be: SIFT, SURF or ORB!");
  }

  // Initialize image properties
  StereoProperties ref_prop;
  hash_matching::StereoProperties::Params image_params;
  image_params.desc_type = desc_type;
  ref_prop.setParams(image_params);

  // Initialize hash
  Hash hash_obj;
  hash_matching::Hash::Params hash_params;
  hash_params.proj_num = proj_num;
  hash_params.features_max_value = features_max_value;
  hash_params.features_min_value = features_min_value;
  hash_obj.setParams(hash_params);

  // Initialize the output
  ostringstream output_csv;

  // Initialize hash distances
  vector< pair<string, vector<float> > > res_table;

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

  // Read the ground truth file
  int total_lc = 0;
  vector< vector<int> > ground_truth;
  ifstream in(gt_file.c_str());
  if (!in) ROS_ERROR("[HashMatching: ] Ground truth file does not exist.");
  for (int x=0; x<(int)v.size(); x++) 
  {
    vector<int> row;
    for (int y=0; y<(int)v.size(); y++)
    {
      int num;
      in >> num;
      row.push_back(num);
    }
    ground_truth.push_back(row);
    int sum_of_elems = accumulate(row.begin(),row.end(),0);
    if (sum_of_elems > 0)
      total_lc++;
  }
  in.close();

  // Create the directory to store the keypoints and descriptors
  string ex_folder = img_dir+"/ex";
  if (fs::is_directory(ex_folder))
    fs::remove_all(ex_folder);
  fs::path dir(ex_folder);
  if (!fs::create_directory(dir))
    ROS_ERROR("[HashMatching:] Impossible to create the execution directory.");

  // Count the overall loop time
  ros::WallTime overall_time_start = ros::WallTime::now();

  // For stereo datasets, gt_tolerance must be twice the gt_tolerance
  if (stereo_dataset)
    gt_tolerance = 2*gt_tolerance;
  
  // Iterate over all images
  bool first = true;
  int found_lc = 0;
  int true_positives = 0;
  int false_positives = 0;
  int gt_offset = 0;
  while (it!=v.end())
  {
    // Check if the directory entry is an directory.
    if (!fs::is_directory(*it)) 
    {
      // Get filename
      string filename = it->filename().string();
      string rawname = getImageIdx(filename);

      // For the stereo dataset, process only left images
      if (stereo_dataset)
      {
        if (boost::lexical_cast<int>(rawname) % 2 == 0) {
          it++;
          continue;
        }
      }

      // Read image
      string path = img_dir+"/"+filename;
      Mat img_ref = imread(path, CV_LOAD_IMAGE_COLOR);
      ref_prop.setImage(img_ref);
      vector<Point2f> ref_points;
      for(int i=0; i<ref_prop.getKp().size(); i++)
        ref_points.push_back(ref_prop.getKp()[i].pt);

      // Save kp and descriptors
      FileStorage fs(ex_folder+"/"+rawname+".yml", FileStorage::WRITE);
      write(fs, "kp", ref_points);
      write(fs, "desc", ref_prop.getDesc());
      fs.release();

      // For the first image initialize the hash object
      if(first)
      {
        if (boost::lexical_cast<int>(rawname) == 1)
          gt_offset = 1;
        ROS_INFO_STREAM("[HashMatching:] GT Offset: " << gt_offset);
        hash_obj.init(ref_prop.getDesc(), proj_orthogonal);
        first = false;
      }

      // Ge the hash for this image
      vector<float> ref_hash = hash_obj.getHash3(ref_prop.getDesc());

      // Save into table
      res_table.push_back(make_pair(filename, ref_hash));

      // Check if table is big enough
      if (res_table.size() <= min_neighbour)
      {
        it++;
        continue;
      }

      // Compute the hash matchings for this image with all other sequence
      vector< pair<string,float> > matchings;
      for (uint k=0; k<res_table.size()-min_neighbour; k++)
      {
        // Hash matching
        vector<float> cur_hash = res_table[k].second;
        float m = match(ref_hash, cur_hash);
        matchings.push_back(make_pair(res_table[k].first, m));
      }

      // Sort the hash matchings
      sort(matchings.begin(), matchings.end(), hash_matching::Utils::sortByDistance);

      // Check for loop closure
      int best_m=0;
      int matches = 0;
      int inliers = 0;
      bool valid = false;
      while (best_m<n_levels)
      {
        // Get the candidate image and compute the descriptors
        if(best_m >= matchings.size())
        {
          best_m = 0;
          break;
        }

        // Loop-closure?
        stringstream ss;
        int img_idx = boost::lexical_cast<int>(getImageIdx(matchings[best_m].first));
        ss << setw(rawname.size()) << setfill('0') << img_idx;
        valid = loopClosure(ref_prop, ex_folder+"/"+ss.str()+".yml", desc_thresh, min_matches, min_inliers, matches, inliers);
        if (valid && !validate) break;

        // Validate the loop closure?
        if (valid && validate)
        {
          // Initialize validation
          bool validate_valid = false;
          int matches_val, inliers_val;

          int off_stereo = 0;
          if (stereo_dataset)
            off_stereo = 1;

          // Loop closure for the previous image 
          stringstream ss;
          int prev_img_idx = img_idx - 1 - off_stereo;
          ss << setw(rawname.size()) << setfill('0') << prev_img_idx;
          validate_valid = loopClosure(ref_prop, ex_folder+"/"+ss.str()+".yml", desc_thresh, min_matches, min_inliers, matches_val, inliers_val);

          if (!validate_valid)
          {
            // Previous validation does not works, try to validate with the next image
            stringstream ss;
            int next_img_idx = img_idx + 1 + off_stereo;
            ss << setw(rawname.size()) << setfill('0') << next_img_idx;
            validate_valid = loopClosure(ref_prop, ex_folder+"/"+ss.str()+".yml", desc_thresh, min_matches, min_inliers, matches_val, inliers_val);
          }

          // If validation, exit. If not, mark as non-valid
          if (validate_valid)
            break;
          else
            valid = false;
        }

        best_m++;
      }

      // Check ground truth
      int tp = 0;
      int fa = 0;
      if (valid)
      {
        found_lc++;

        int cur_img_idx = boost::lexical_cast<int>(getImageIdx(matchings[best_m].first));
        int gt_valid = 0;
        for (int i=0; i<2*gt_tolerance+1; i++)
        {
          int idx = cur_img_idx - gt_tolerance + i;
          if (idx<0) idx = 0;
          gt_valid += ground_truth[boost::lexical_cast<int>(rawname) - gt_offset][idx];
        }
        
        if(gt_valid >= 1)
        {
          true_positives++;
          tp = 1;
        }
        if(gt_valid == 0)
        {
          false_positives++;
          fa = 1;
        }
      }

      // Log
      if(best_m >= matchings.size()) best_m = 0;
      ROS_INFO_STREAM( rawname << " cl with " << getImageIdx(matchings[best_m].first) << ": " << valid << " (" << matches << "/" << inliers << "). " << tp << "|" << fa);
      output_csv << rawname << "," << getImageIdx(matchings[best_m].first) << "," << valid << "," << matches << "," << inliers << endl;

    }
    // Next directory entry
    it++;
  }

  // Remove the temporal directory
  if (fs::is_directory(ex_folder))
    fs::remove_all(ex_folder);

  // Stop time
  ros::WallDuration overall_time = ros::WallTime::now() - overall_time_start;

  // Compute precision and recall
  int false_negatives = total_lc - found_lc;
  double precision = round( 100 * true_positives / (true_positives + false_positives) );
  double recall = round( 100 * true_positives / (true_positives + false_negatives) );

  // Print the results
  ROS_INFO_STREAM("TOTAL #LC: " << total_lc);
  ROS_INFO_STREAM("FOUND #LC: " << found_lc);
  ROS_INFO_STREAM("#TP: " << true_positives);
  ROS_INFO_STREAM("#FP: " << false_positives);
  ROS_INFO_STREAM("PRECISION: " << precision << "%");
  ROS_INFO_STREAM("RECALL: " << recall << "%");
  ROS_INFO_STREAM("TOTAL EXECUTION TIME: " << overall_time.toSec() << " sec.");

  // Save data into file
  string out_file = files_path + desc_type + "_" + toString(proj_num) + ".txt";
  fstream f_out(out_file.c_str(), ios::out | ios::trunc);
  f_out << output_csv.str();
  f_out.close();

  // Close the node
  ros::shutdown();
}

// Hash matching functions
float hash_matching::TestBase::match(vector<float> hash_1, vector<float> hash_2)
{
  ROS_ASSERT(hash_1.size() == hash_2.size()); // Sanity check

  // Compute the distance
  float sum = 0.0;
  for (uint i=0; i<hash_1.size(); i++)
    sum += fabs(hash_1[i] - hash_2[i]);
  
  return sum;
}

// Converts number to string
template <typename T> string hash_matching::TestBase::toString( const T& n )
{
  ostringstream stm;
  stm << n;
  return stm.str();
}

// Return the image index from the filename
string hash_matching::TestBase::getImageIdx(string filename)
{
  int point_idx = filename.find_last_of(".");
  int bar_idx = filename.find_last_of("_");
  return filename.substr(bar_idx+1, point_idx-bar_idx-1);
}

// True if images close loop
bool hash_matching::TestBase::loopClosure(StereoProperties ref_prop,
                                          string cur_filename,
                                          double desc_thresh, 
                                          int min_matches, 
                                          int min_inliers,
                                          int &matches,
                                          int &inliers)
{
  // Initialize outputs
  matches = 0;
  inliers = 0;

  // Sanity check
  if ( !fs::exists(cur_filename) ) return false;

  // Get the image keypooints and descriptors
  FileStorage fs; 
  fs.open(cur_filename, FileStorage::READ);
  if (!fs.isOpened()) ROS_ERROR("[HashMatching:] Failed to open the image keypoints and descriptors.");
  vector<Point2f> cur_kp;
  Mat cur_desc;
  fs["kp"] >> cur_kp;
  fs["desc"] >> cur_desc;
  fs.release();

  // Descriptors crosscheck matching
  vector<DMatch> desc_matches;
  Mat match_mask;
  hash_matching::Utils::crossCheckThresholdMatching(ref_prop.getDesc(), 
                                                    cur_desc, 
                                                    desc_thresh, 
                                                    match_mask, desc_matches);
  matches = (int)desc_matches.size();

  // Check matches size
  if (matches < min_matches)
    return false;

  // Get the matched keypoints
  vector<KeyPoint> ref_kp = ref_prop.getKp();
  vector<Point2f> ref_points;
  vector<Point2f> cur_points;
  for(int i=0; i<matches; i++)
  {
    ref_points.push_back(ref_kp[desc_matches[i].queryIdx].pt);
    cur_points.push_back(cur_kp[desc_matches[i].trainIdx]);
  }

  // Check the epipolar geometry
  Mat status;
  Mat F = findFundamentalMat(ref_points, cur_points, FM_RANSAC, 1, 0.999, status);

  // Is the fundamental matrix valid?
  Scalar f_sum_parts = cv::sum(F);
  float f_sum = (float)f_sum_parts[0] + (float)f_sum_parts[1] + (float)f_sum_parts[2];
  if (f_sum < 1e-3)
    return false;

  // Check inliers size
  inliers = (int)cv::sum(status)[0];
  if (inliers < min_inliers)
    return false;

  // If we arrive here, there is a loop closure.
  return true;
}