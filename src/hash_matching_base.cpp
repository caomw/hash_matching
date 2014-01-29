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
  int num_hyperplanes, bucket_width, bucket_height, bucket_max, best_n, features_max_value, N_levels;
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
  nh_private_.getParam("features_max_value", features_max_value);
  nh_private_.getParam("N_levels", N_levels);
  

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
  image_params.features_max_value=features_max_value;
  image_params.N_levels=N_levels;
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

  // Initialize the outputs
  ostringstream output_csv;
  ostringstream output_csv1;
  ostringstream output_csv2;
  ostringstream output_csv3;
  ostringstream output_csv4;


  // Read the template image and extract kp and descriptors
  Mat img_temp = imread(ref_path, CV_LOAD_IMAGE_COLOR);
  ref_prop.setImage(img_temp);
  ref_prop.computeHyperplanes();
  ref_prop.computeHash(); // compute hashes of reference frame
  
  // Loop directory images
  typedef std::vector<boost::filesystem::path> vec; // define vec as a type vector of file system paths 
  vec v; // create new vec called v
  copy(
        boost::filesystem::directory_iterator(img_dir), 
        boost::filesystem::directory_iterator(),
        back_inserter(v)
      ); // define a fs iterator pointing to img_dir and send all files to v

  sort(v.begin(), v.end()); // sort v : default sort by alphabetic order. 
  
 // fs::directory_iterator it(img_dir);
 // fs::directory_iterator end;
  
  //vector< pair<double,string> > dists;
  //vector< pair<double,string> > dists1;
  //vector< pair<double,string> > dists2;
  //vector< pair<double,string> > dists3;
  //vector< pair<double,string> > dists4;
  vector< trio > dists;
  vector< trio > dists1;
  vector< trio > dists2;
  vector< trio > dists3;
  vector< trio > dists4;
  vector< trio > dists5;

  vec::const_iterator it(v.begin()); // define an iterator over v, starting at the 1st element. 
  // it is actually a pointer, so a reference to its methods needs a "->"
  while (it!=v.end())
  {
    // Check if the directory entry is an empty directory.
    if (!fs::is_directory(*it)) 
    {
      // Read image
      string path = img_dir + "/" + it->filename().string();
      ROS_INFO_STREAM("current_image" << it->filename().string());
      Mat img_cur = imread(path, CV_LOAD_IMAGE_COLOR);
      if (path==ref_path){ // if the reference image is found, terminate. 
        ROS_INFO_STREAM(it->filename().string() << "He encontrado la imagen actual, salgo");
        break;
      }
      cur_prop.setImage(img_cur);
      cur_prop.setHyperplanes(ref_prop.getCentroid(), ref_prop.getH(), ref_prop.getDelta());

      // Crosscheck feature matching
      vector<DMatch> matches; // feature matchings between two images
      Mat match_mask;
      hash_matching::OpencvUtils::crossCheckThresholdMatching(ref_prop.getDesc(), 
                                                              cur_prop.getDesc(), 
                                                              desc_thresh, 
                                                              match_mask, matches);
      // Compute current image hash
      cur_prop.computeHash();
      // Compare image hashes
   //   ROS_INFO_STREAM(it->path().filename().string() << "has1");
      double hash_matching1 = match(ref_prop.getHash1(), cur_prop.getHash1());
      // first hash: vector with number of feature/region
   //   ROS_INFO_STREAM(it->path().filename().string() << "has2");
      double hash_matching2 = match(ref_prop.getHash2(), cur_prop.getHash2());
      // phase between bin centroid and global centroid
  //    ROS_INFO_STREAM(it->path().filename().string() << "has3");
      double hash_matching3 = match(ref_prop.getHash3(), cur_prop.getHash3());
      // module of the distance between bin centroid and global centroid
  //    ROS_INFO_STREAM(it->path().filename().string() << "has4");
      double hash_matching4 = match(ref_prop.getHash4(), cur_prop.getHash4());
      double hash_matching5 = match(ref_prop.getHash5(), cur_prop.getHash5());

      // dispersion (variance) of features with the same bin with respect their centroid.
     // double matching = log10(hash_matching1*hash_matching2*hash_matching3*hash_matching4*hash_matching5); // module of the 4 component hash
      //double matching = log10(hash_matching1*hash_matching2*hash_matching3); // module of the 4 component hash
      
      // double a1=0.5;
      // double a2=0.5;
      // double a3=0.5;
      // double a4=0.01;
      // double a5=1.5;

      double a1=0.05;
      double a2=0.05;
      double a3=0.05;
      double a4=0.01;
      double a5=1.75;

      double matching = ((hash_matching1*a1)+(hash_matching2*a2)+(hash_matching3*a3)+(hash_matching4*a4)+(hash_matching5*a5));

      ROS_INFO_STREAM("hash matchings" << hash_matching1 << ";" << hash_matching2 << ";" << hash_matching3 << ";" << hash_matching4 << ";" << hash_matching5);
      
      if (isfinite(matching))
      {
        // Log image hash matching vs feature matchings
        ROS_INFO_STREAM(it->filename().string() << " -> matching: " <<
          matching << " | Desc. Matches: " << (int)matches.size());

        output_csv << matching << "," << matches.size() << endl; 
        output_csv1 << hash_matching1 << "," << matches.size() << endl; 
        output_csv2 << hash_matching2 << "," << matches.size() << endl; 
        output_csv3 << hash_matching3 << "," << matches.size() << endl; 
        output_csv4 << hash_matching4 << "," << matches.size() << endl; 
        // Save hashes into vectors
        trio trio1,trio2,trio3,trio4,trio5, trio6;
        trio1.hashmatching=matching;
        trio1.featurematchings=(int)matches.size();
        trio1.image=it->filename().string();

        trio2.hashmatching=hash_matching1;
        trio2.featurematchings=(int)matches.size();
        trio2.image=it->filename().string();
        
        trio3.hashmatching=hash_matching2;
        trio3.featurematchings=(int)matches.size();
        trio3.image=it->filename().string();
        
        trio4.hashmatching=hash_matching3;
        trio4.featurematchings=(int)matches.size();
        trio4.image=it->filename().string();

        trio5.hashmatching=hash_matching4;
        trio5.featurematchings=(int)matches.size();
        trio5.image=it->filename().string();

        trio6.hashmatching=hash_matching5;
        trio6.featurematchings=(int)matches.size();
        trio6.image=it->filename().string();

        dists.push_back(trio1);
        dists1.push_back(trio2);
        dists2.push_back(trio3);
        dists3.push_back(trio4);
        dists4.push_back(trio5);
        dists5.push_back(trio6);

        // dists1.push_back(make_pair(hash_matching1, it->path().filename().string()));
        // dists2.push_back(make_pair(hash_matching2, it->path().filename().string()));
        // dists3.push_back(make_pair(hash_matching3, it->path().filename().string()));
        // dists4.push_back(make_pair(hash_matching4, it->path().filename().string()));
      }
    }
    // Next directory entry
    it++;
  }

  // Save data into file
  string out_file;
  out_file = files_path + "output.txt";

  string out_file1;
  out_file1 = files_path + "output1.txt";

  string out_file2;
  out_file2 = files_path + "output2.txt";

  string out_file3;
  out_file3 = files_path + "output3.txt";

  string out_file4;
  out_file4 = files_path + "output4.txt";
  fstream f_out(out_file.c_str(), ios::out | ios::trunc);
  f_out << output_csv.str();
  f_out.close();

  fstream f_out1(out_file1.c_str(), ios::out | ios::trunc);
  f_out1 << output_csv1.str();
  f_out1.close();

  fstream f_out2(out_file2.c_str(), ios::out | ios::trunc);
  f_out2 << output_csv2.str();
  f_out2.close();

  fstream f_out3(out_file3.c_str(), ios::out | ios::trunc);
  f_out3 << output_csv3.str();
  f_out3.close();

  fstream f_out4(out_file4.c_str(), ios::out | ios::trunc);
  f_out4 << output_csv4.str();
  f_out4.close();
  // Sort distances vector
  sort(dists.begin(), dists.end(), hash_matching::OpencvUtils::sorttrioByDistance);
  sort(dists1.begin(), dists1.end(), hash_matching::OpencvUtils::sorttrioByDistance);
  sort(dists2.begin(), dists2.end(), hash_matching::OpencvUtils::sorttrioByDistance);
  sort(dists3.begin(), dists3.end(), hash_matching::OpencvUtils::sorttrioByDistance);
  sort(dists4.begin(), dists4.end(), hash_matching::OpencvUtils::sorttrioByDistance);
  sort(dists5.begin(), dists5.end(), hash_matching::OpencvUtils::sorttrioByDistance);
  // Show results
  for(int i=0; i<best_n; i++)
  {
    ROS_INFO_STREAM("BEST MATCHING:   " << dists[i].image << " (" << dists[i].hashmatching << ")" << " (" << dists[i].featurematchings << ")");
    ROS_INFO_STREAM("BEST MATCHING 1: " << dists1[i].image << " (" << dists1[i].hashmatching << ")" << "(" << dists1[i].featurematchings << ")");
    ROS_INFO_STREAM("BEST MATCHING 2: " << dists2[i].image << " (" << dists2[i].hashmatching << ")" << "(" << dists2[i].featurematchings << ")");
    ROS_INFO_STREAM("BEST MATCHING 3: " << dists3[i].image << " (" << dists3[i].hashmatching << ")" << "(" << dists3[i].featurematchings << ")");
    ROS_INFO_STREAM("BEST MATCHING 4: " << dists4[i].image << " (" << dists4[i].hashmatching << ")" << "(" << dists4[i].featurematchings << ")");
    ROS_INFO_STREAM("BEST MATCHING 5: " << dists5[i].image << " (" << dists5[i].hashmatching << ")" << "(" << dists5[i].featurematchings << ")");
  }
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
  //  ROS_INFO_STREAM("hash_1;hash_2;" << (double)hash_1[i] << ";" << (double)hash_2[i] );
  }
  return sqrt(sum); // euclidean distance
}

double hash_matching::HashMatchingBase::match(vector<double> hash_1, vector<double> hash_2)
{
  // Sanity check
  ROS_ASSERT(hash_1.size() == hash_2.size());

  double sum = 0.0;
  for (uint i=0; i<hash_1.size(); i++) // the module of the difference
  {
    sum += pow((double)hash_1[i] - (double)hash_2[i], 2.0);
  //  ROS_INFO_STREAM("hash_1;hash_2;" << (double)hash_1[i] << ";" << (double)hash_2[i] );
  }
  return sqrt(sum);
}