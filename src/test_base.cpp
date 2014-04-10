#include <ros/package.h>
#include <boost/shared_ptr.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>
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
  double desc_thresh, max_avg_err;
  bool proj_orthogonal, validate;
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
  nh_private_.getParam("max_avg_err", max_avg_err);
  nh_private_.param("validate", validate, false);
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
  cout << "  max_avg_err      = " << max_avg_err << endl;
  cout << "  validate         = " << validate << endl;
  cout << "  gt_tolerance     = " << gt_tolerance << endl;

  // Files path sanity check
  if (files_path[files_path.length()-1] != '/')
    files_path += "/";

  // Sanity checks
  if (!fs::exists(img_dir) || !fs::is_directory(img_dir)) 
  {
    ROS_ERROR_STREAM("[HashMatching:] The image directory does not exists: " << 
                     img_dir);
    return;
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
    ROS_ERROR("[HashMatching: ] Descriptor type must be: SIFT, SURF or ORB!");
  }

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
  vector< vector<int> > gt;
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
    gt.push_back(row);
  }
  in.close();

  // Count the number of total loops.
  int total_lc = 0;
  for (int x=0; x<(int)v.size(); x++) 
  {
    for (int y=0; y<(int)v.size(); y++)
    {
      if (gt[x][y] == 1)
      {
        total_lc++;
        break;
      }
    }
  }
  ROS_INFO_STREAM("TOTAL #LC: " << total_lc);

  // Count the overall loop time
  ros::WallTime overall_time_start = ros::WallTime::now();
  
  // Iterate over all images
  bool first = true;
  int found_lc = 0;
  int true_positives = 0;
  int false_alarm = 0;
  while (it!=v.end())
  {
    // Check if the directory entry is an directory.
    if (!fs::is_directory(*it)) 
    {
      // Get filename
      string filename = it->filename().string();
      string rawname = getImageIdx(filename);
      int ref_img_idx = boost::lexical_cast<int>(rawname);

      // Read image
      string path = img_dir + "/" + filename;
      Mat img_ref = imread(path, CV_LOAD_IMAGE_COLOR);
      ref_prop.setImage(img_ref);

      // For the first image initialize the hash object
      if(first)
      {
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
      double avg_err = 0.0;
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
        valid = loopClosure(ref_prop, cur_prop, img_dir+"/"+matchings[best_m].first, desc_thresh, min_matches, min_inliers, max_avg_err, matches, inliers, avg_err);
        if (valid && !validate) break;

        // Validate the loop closure?
        if (valid && validate)
        {
          // Initialize validation
          bool validate_valid = false;

          // Loop closure for the previous image 
          int prev_img_idx = boost::lexical_cast<int>(getImageIdx(matchings[best_m].first)) - 1;
          string prev_img_file = buildImageName(prev_img_idx, matchings[best_m].first);
          string tmp_path = img_dir + "/" + prev_img_file;
          if(fs::exists(tmp_path))
            validate_valid = loopClosure(ref_prop, cur_prop, img_dir+"/"+prev_img_file, desc_thresh, min_matches, min_inliers, max_avg_err, matches, inliers, avg_err);

          ROS_INFO_STREAM("VALIDATING PREV: " << validate_valid);
          if (!validate_valid)
          {
            // Previous validation does not works, try to validate with the next image
            int prev_img_idx = boost::lexical_cast<int>(getImageIdx(matchings[best_m].first)) + 1;
            string prev_img_file = buildImageName(prev_img_idx, matchings[best_m].first);
            string tmp_path = img_dir + "/" + prev_img_file;
            if(fs::exists(tmp_path))
              validate_valid = loopClosure(ref_prop, cur_prop, img_dir+"/"+prev_img_file, desc_thresh, min_matches, min_inliers, max_avg_err, matches, inliers, avg_err);
            ROS_INFO_STREAM("VALIDATING NEXT: " << validate_valid);
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
          gt_valid += gt[ref_img_idx][idx];
        }
        
        if(gt_valid >= 1)
        {
          true_positives++;
          tp = 1;
        }
        if(gt_valid == 0)
        {
          false_alarm++;
          fa = 1;
        }
      }

      // Log
      if(best_m >= matchings.size()) best_m = 0;
      ROS_INFO_STREAM( rawname << " cl with " << getImageIdx(matchings[best_m].first) << ": " << valid << " (" << matches << "/" << inliers << "/" << avg_err << "). " << tp << "|" << fa);
      output_csv << rawname << "," << getImageIdx(matchings[best_m].first) << "," << valid << "," << matches << "," << inliers << "," << avg_err << endl;

    }
    // Next directory entry
    it++;
  }

  // Print the overall time
  ros::WallDuration overall_time = ros::WallTime::now() - overall_time_start;
  ROS_INFO_STREAM("TOTAL #LC: " << total_lc);
  ROS_INFO_STREAM("FOUND #LC: " << found_lc);
  ROS_INFO_STREAM("#TP: " << true_positives);
  ROS_INFO_STREAM("#FA: " << false_alarm);
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

string hash_matching::TestBase::buildImageName(int img_idx, string filename_template)
{
  int point_idx = filename_template.find_last_of(".");
  int bar_idx = filename_template.find_last_of("_");
  string start = filename_template.substr(0, bar_idx+1);

  stringstream ss;
  ss << setw(6) << setfill('0') << img_idx;
  string s = start + ss.str() + filename_template.substr(point_idx, 4);
  return s;
}

// True if images close loop
bool hash_matching::TestBase::loopClosure(StereoProperties ref_prop, 
                                          StereoProperties cur_prop, 
                                          string cur_filename,
                                          double desc_thresh, 
                                          int min_matches, 
                                          int min_inliers,
                                          double max_avg_err,
                                          int &matches,
                                          int &inliers,
                                          double &avg_err)
{
  // Initialize outputs
  matches = 0;
  inliers = 0;
  avg_err = 0.0;

  // Get the image and compute descriptors
  Mat tmp_img = imread(cur_filename, CV_LOAD_IMAGE_COLOR);
  cur_prop.setImage(tmp_img);

  // Descriptors crosscheck matching
  vector<DMatch> desc_matches;
  Mat match_mask;
  hash_matching::Utils::crossCheckThresholdMatching(ref_prop.getDesc(), 
                                                    cur_prop.getDesc(), 
                                                    desc_thresh, 
                                                    match_mask, desc_matches);
  matches = (int)desc_matches.size();

  // Check matches size
  if (matches < min_matches)
    return false;

  // Get the matched keypoints
  vector<KeyPoint> ref_kp = ref_prop.getKp();
  vector<KeyPoint> cur_kp = cur_prop.getKp();
  vector<Point2f> ref_points;
  vector<Point2f> cur_points;
  for(int i=0; i<matches; i++)
  {
    ref_points.push_back(ref_kp[desc_matches[i].queryIdx].pt);
    cur_points.push_back(cur_kp[desc_matches[i].trainIdx].pt);
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

  // Fundamental matrix quality check (line 260 of https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/c/stereo_calib.cpp?rev=2614)
  // First, select the inliers
  vector<Point2f> ref_points_in;
  vector<Point2f> cur_points_in;
  for(int i=0; i<status.rows; i++)
  {
    if((int)status.at<uchar>(0, i) == 1)
    {
      ref_points_in.push_back(ref_points[i]);
      cur_points_in.push_back(cur_points[i]);
    }
  }
  // Second compute the average error
  vector<Vec3f> lines1, lines2;
  computeCorrespondEpilines(ref_points_in, 1, F, lines1);
  computeCorrespondEpilines(cur_points_in, 2, F, lines2);
  for(int i=0; i<inliers; i++)
  {
    double err =  fabs(ref_points_in[i].x*lines2[i][0] +
                       ref_points_in[i].y*lines2[i][1] + lines2[i][2]) +
                  fabs(cur_points_in[i].x*lines1[i][0] +
                       cur_points_in[i].y*lines1[i][1] + lines1[i][2]);
    avg_err += err;
  }
  avg_err /= inliers;

  // Check reprojection error
  if (avg_err >= max_avg_err)
    return false;

  // If we arrive here, there is a loop closure.
  return true;
}