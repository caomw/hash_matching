#include "hash_matching_base.h"
#include "stereo_properties.h"
#include "opencv_utils.h"
#include <boost/shared_ptr.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/lambda/bind.hpp>
#include <time.h>
#include "LSH_functions.h" 

namespace fs=boost::filesystem;
using namespace boost::lambda;

hash_matching::HashMatchingBase::HashMatchingBase(
  ros::NodeHandle nh, ros::NodeHandle nhp) : nh_(nh), nh_private_(nhp)
{
  // Load parameters
  string ref_path, img_dir, desc_type, files_path;
  double desc_thresh;
  int eigen_dim, num_hyperplanes, bucket_width, bucket_height, bucket_max, best_n, features_max_value, N_levels, tolerance;
  nh_private_.param("ref_path", ref_path, std::string(""));
  nh_private_.param("img_dir", img_dir, std::string(""));
  nh_private_.param("desc_type", desc_type, std::string("SIFT"));
  nh_private_.getParam("desc_thresh", desc_thresh);
  nh_private_.getParam("best_n", best_n);
  nh_private_.getParam("eigen_dim", eigen_dim);
  nh_private_.getParam("num_hyperplanes", num_hyperplanes);
  nh_private_.getParam("bucket_width", bucket_width);
  nh_private_.getParam("bucket_height", bucket_height);
  nh_private_.getParam("bucket_max", bucket_max);
  nh_private_.param("files_path", files_path, std::string("/home/user"));
  nh_private_.getParam("features_max_value", features_max_value);
  nh_private_.getParam("N_levels", N_levels);
  nh_private_.getParam("tolerance", tolerance);
  

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

  // Log for hash tables
  string hash_tables;
  hash_tables = files_path + "hash_tables.txt";
  string hash1_compare, hash2_compare, hash3_compare, hash4_compare, hash5_compare, hash6_compare;
  hash1_compare = files_path + "hash1_compare.txt";
  hash2_compare = files_path + "hash2_compare.txt";
  hash3_compare = files_path + "hash3_compare.txt";
  hash4_compare = files_path + "hash4_compare.txt";
  hash5_compare = files_path + "hash5_compare.txt";
  hash6_compare = files_path + "hash5_compare.txt";
  
  ostringstream hash_indexes, hash_comparison1, hash_comparison2, hash_comparison3, hash_comparison4, hash_comparison5, hash_comparison6;

  // Count how many images are into the directory
  fs::path img_path(img_dir);

  int cnt = count_if(
      fs::directory_iterator(img_path),
      fs::directory_iterator(),
      bind( static_cast<bool(*)(const fs::path&)>(fs::is_regular_file), 
      bind( &fs::directory_entry::path,  boost::lambda::_1 ) ) );
  Int32T hashTableSize = cnt; 

  // Read the template image and extract kp and descriptors
  Mat img_temp = imread(ref_path, CV_LOAD_IMAGE_COLOR);
  ref_prop.setImage(img_temp);
  ROS_INFO_STREAM("Reference Keypoints Size: " << ref_prop.getKp().size());
  uint seedu= ((uint)time(NULL)); // projections: the same seed for the ref. image and for the current images makes the random vector to be the same. 
  ref_prop.computeHyperplanes();
  ref_prop.computeHash(); // compute hashes (1:5) of reference frame
  // compute the hash based on projections of features using a scalar vector with random components
  vector<double> ref_hash_projections = ref_prop.ComputeProjections(ref_prop.getDesc(), seedu);


  // until now, we have computed the 5 hashes (number of features/inter-plane regions, angle and norm 
  // of each inter-plane centroid with respect to the global centroid, variance of features of a certain region 
  // with respect the regioon centroid and feature discretization hytogram ). From now on, we call these 
  // hashes as the bucket functions associated to every image. So, every image has associated 5 vectors, which 
  // are called its buckets. We will use a hashtable for each bucket function type. 

  // for the LSH process we need to know the exact sice of the HashTable, and this is only available when all images 
  // have been processed. 
  vector <float> r1; 
  vector <float> r2; 
  vector <float> r3; 
  vector <float> r4; 
  vector <float> r5; 
  vector <float> r6; 
  vector <Uns32T> hash_table1, hash_table2, hash_table3, hash_table4, hash_table5, hash_table6;

  // Hash indexs for the reference image
  // compute random vectors 
  uint seedh = ((uint)time(NULL))*1e-08;
  r1 = compute_random_vector(seedh, ref_prop.getHash1().size()); // different seeds for the LSH computation
  r2 = compute_random_vector(seedh+1, ref_prop.getHash2().size());
  r3 = compute_random_vector(seedh+2, ref_prop.getHash3().size());
  r4 = compute_random_vector(seedh+3, ref_prop.getHash4().size());
  r5 = compute_random_vector(seedh+4, ref_prop.getHash5().size());
  r6 = compute_random_vector(seedh+5, ref_prop.getHash6().size());

  // reference image --> generate the hash functions (indexes in the hash tables 1,2,3,4,5)
  // every hash function gives the index of a certain bucket of a certain image inside its corresponding 
  // hash table. This index is computed using random numbers. 
  vector<double> r; // index on hash table 1 
  for (int i=0; i<ref_prop.getHash1().size(); i++)
    r.push_back((double)ref_prop.getHash1()[i] );;

  Uns32T hashfunction1_ref = computeUHashFunction(r1,r,ref_prop.getHash1().size(), UH_PRIME_DEFAULT, hashTableSize);
  Uns32T hashfunction2_ref = computeUHashFunction(r2,ref_prop.getHash2(),ref_prop.getHash2().size(), UH_PRIME_DEFAULT, hashTableSize);
  Uns32T hashfunction3_ref = computeUHashFunction(r3,ref_prop.getHash3(),ref_prop.getHash3().size(), UH_PRIME_DEFAULT, hashTableSize);
  Uns32T hashfunction4_ref = computeUHashFunction(r4,ref_prop.getHash4(),ref_prop.getHash4().size(), UH_PRIME_DEFAULT, hashTableSize);
  r.clear();
  for (int i=0; i<ref_prop.getHash5().size(); i++)
    r.push_back((double)ref_prop.getHash5()[i] );
  Uns32T hashfunction5_ref = computeUHashFunction(r5,r,ref_prop.getHash5().size(), UH_PRIME_DEFAULT, hashTableSize);
  Uns32T hashfunction6_ref = computeUHashFunction(r6,ref_prop.getHash6(),ref_prop.getHash6().size(), UH_PRIME_DEFAULT, hashTableSize);
 
  hash_indexes << "hash indexes ref. image: " << hashfunction1_ref << "," << hashfunction2_ref << ";" << hashfunction3_ref << ";" << hashfunction4_ref << ";" << hashfunction5_ref << endl; 
  
  // Define struct
  trio comp;
  vector <trio> comparison1, comparison2, comparison3, comparison4, comparison5, comparison6;
  
  // Loop directory images
  typedef std::vector<boost::filesystem::path> vec; // define vec as a type vector of file system paths 
  vec v; // create new vec called v
  copy(
        boost::filesystem::directory_iterator(img_dir), 
        boost::filesystem::directory_iterator(),
        back_inserter(v)
      ); // define a fs iterator pointing to img_dir and send all files to v

  sort(v.begin(), v.end()); // sort v : default sort by alphabetic order. 

  vec::const_iterator it(v.begin()); // define an iterator over v, starting at the 1st element. 
  // it is actually a pointer, so a reference to its methods needs a "->"
  // store the 5 buckets in a vector of vectors 
  
  // Open log files
  fstream h_out(hash_tables.c_str(), ios::out | ios::trunc);
  fstream h1_out(hash1_compare.c_str(), ios::out | ios::trunc);
  fstream h2_out(hash2_compare.c_str(), ios::out | ios::trunc);
  fstream h3_out(hash3_compare.c_str(), ios::out | ios::trunc);
  fstream h4_out(hash4_compare.c_str(), ios::out | ios::trunc);
  fstream h5_out(hash5_compare.c_str(), ios::out | ios::trunc);
  fstream h6_out(hash6_compare.c_str(), ios::out | ios::trunc);
  int count=0;
  // Iterate
  while (it!=v.end()) // for every key image
  { 
    // Check if the directory entry is an empty directory. The content of the it is a filename
    if (!fs::is_directory(*it)) 
    {
      // Filename
      count++; 
      string filename = it->filename().string();
      int lastindex = filename.find_last_of(".");
      string rawname = filename.substr(0, lastindex);
      ROS_INFO_STREAM("Processing image: " << rawname);
      
      // Read image
      string path = img_dir + "/" + filename;
      Mat img_cur = imread(path, CV_LOAD_IMAGE_COLOR);
      if (path==ref_path){ // if the reference image is found, terminate. 
        ROS_INFO_STREAM(it->filename().string() << "He encontrado la imagen actual, salgo");
        break;
      }
      cur_prop.setImage(img_cur);
      cur_prop.setHyperplanes(ref_prop.getCentroid(), ref_prop.getH(), ref_prop.getDelta());


      // Crosscheck feature matching
      vector<DMatch> matches;
      Mat match_mask;
      hash_matching::OpencvUtils::crossCheckThresholdMatching(ref_prop.getDesc(), 
                                                              cur_prop.getDesc(), 
                                                              desc_thresh, 
                                                              match_mask, matches);

      // Compute current image hash
      cur_prop.computeHash();
      vector<double> cur_hash_projections = cur_prop.ComputeProjections(cur_prop.getDesc(), seedu);

      // Compare image hashes
      double hash_matching1 = match(ref_prop.getHash1(), cur_prop.getHash1());
      // first hash: vector with number of feature/region
      double hash_matching2 = match(ref_prop.getHash2(), cur_prop.getHash2());
      // phase between bin centroid and global centroid
      double hash_matching3 = match(ref_prop.getHash3(), cur_prop.getHash3());
      // module of the distance between bin centroid and global centroid
      double hash_matching4 = match(ref_prop.getHash4(), cur_prop.getHash4());
      // dispersion (variance) of features with the same bin with respect their centroid.
      double hash_matching5 = match(ref_prop.getHash5(), cur_prop.getHash5());
      // hystogram of discretized feature components
      double hash_matching6 = match(ref_hash_projections, cur_hash_projections);
      // projections of features on a random direction (escalar product)
      
      ROS_INFO_STREAM("matching 6" << hash_matching6);

      vector<double> b1,b5; // index on hash table 1,5 
      for (int c=0; c<cur_prop.getHash1().size(); c++)
        b1.push_back((double)cur_prop.getHash1()[c] ); // convert bucket1 fro int to double; hashfunction needs (double) vector
      for (int c=0; c<cur_prop.getHash5().size(); c++)
        b5.push_back((double)cur_prop.getHash5()[c] ); // convert bucket5 fro int to double; hashfunction needs (double) vector

      Uns32T hashfunction1_cur = computeUHashFunction(r1, b1, b1.size(), UH_PRIME_DEFAULT, hashTableSize);
      Uns32T hashfunction2_cur = computeUHashFunction(r2, cur_prop.getHash2(), cur_prop.getHash2().size(), UH_PRIME_DEFAULT, hashTableSize);
      Uns32T hashfunction3_cur = computeUHashFunction(r3, cur_prop.getHash3(), cur_prop.getHash3().size(), UH_PRIME_DEFAULT, hashTableSize);
      Uns32T hashfunction4_cur = computeUHashFunction(r4, cur_prop.getHash4(), cur_prop.getHash4().size(), UH_PRIME_DEFAULT, hashTableSize);
      Uns32T hashfunction5_cur = computeUHashFunction(r5, b5, b5.size(), UH_PRIME_DEFAULT, hashTableSize);
      Uns32T hashfunction6_cur = computeUHashFunction(r6,cur_prop.getHash6(), cur_prop.getHash6().size(), UH_PRIME_DEFAULT, hashTableSize);
      
      hash_table1.push_back(hashfunction1_cur); // store the hash index in 5 different tables.
      hash_table2.push_back(hashfunction2_cur);
      hash_table3.push_back(hashfunction3_cur);
      hash_table4.push_back(hashfunction4_cur);
      hash_table5.push_back(hashfunction5_cur); // will have as many hash index as the number of  key images
      hash_table6.push_back(hashfunction6_cur);

      // Log
      hash_indexes << rawname << ";" << hashfunction1_cur << ";" << hashfunction2_cur << ";" << hashfunction3_cur << ";" << hashfunction4_cur << ";" << hashfunction5_cur << ";" << hashfunction6_cur << ";" << (int)matches.size() << endl; 

      if ( (hashfunction1_cur <= (hashfunction1_ref+tolerance) ) && (hashfunction1_cur >= (hashfunction1_ref-tolerance) ) )
      {
        int diff1 = abs((int)hashfunction1_cur - (int)hashfunction1_ref);
        comp.featurematchings = (int)matches.size();
        comp.hashmatching = (double)diff1; 
        comp.image = rawname;
        comparison1.push_back(comp);
        hash_comparison1 << comp.image << " " << diff1 << " " << comp.featurematchings << endl; // save in a txt file
      }
      if ( (hashfunction2_cur <= (hashfunction2_ref+tolerance) ) && (hashfunction2_cur >= (hashfunction2_ref-tolerance) ) )
      {
        int diff2 = abs((int)hashfunction2_cur-(int)hashfunction2_ref);
        comp.featurematchings = (int)matches.size(); 
        comp.hashmatching = (double)diff2; 
        comp.image = rawname;
        comparison2.push_back(comp);
        hash_comparison2 << comp.image << " " << diff2 << " " << comp.featurematchings << endl;
      }
      if ( (hashfunction3_cur <= (hashfunction3_ref+tolerance) ) && (hashfunction3_cur >= (hashfunction3_ref-tolerance) ) )
      {
        int diff3 = abs((int)hashfunction3_cur-(int)hashfunction3_ref);
        comp.featurematchings = (int)matches.size();
        comp.hashmatching = (double)diff3; 
        comp.image = rawname;
        comparison3.push_back(comp);
        hash_comparison3 << comp.image << " " << diff3 << " " << comp.featurematchings << endl;
      }
      if ( (hashfunction4_cur <= (hashfunction4_ref+tolerance) ) && (hashfunction4_cur >= (hashfunction4_ref-tolerance) ) )
      {
        int diff4 = abs((int)hashfunction4_cur-(int)hashfunction4_ref);
        comp.featurematchings = (int)matches.size();
        comp.hashmatching = (double)diff4; 
        comp.image = rawname;
        comparison4.push_back(comp);
        hash_comparison4 << comp.image << " " << diff4 << " " << comp.featurematchings << endl;
      }
      if ( (hashfunction5_cur <= (hashfunction5_ref+tolerance) ) && (hashfunction5_cur >= (hashfunction5_ref-tolerance) ) )
      {
        int diff5 = abs((int)hashfunction5_cur-(int)hashfunction5_ref);
        comp.featurematchings = (int)matches.size(); 
        comp.hashmatching = (double)diff5; 
        comp.image = rawname;
        comparison5.push_back(comp);
        hash_comparison5 << comp.image << " " << diff5 << " " << comp.featurematchings << endl;
      }
      if ( (hashfunction6_cur <= (hashfunction6_ref+tolerance) ) && (hashfunction6_cur >= (hashfunction6_ref-tolerance) ) )
      {
        int diff6 = abs((int)hashfunction6_cur-(int)hashfunction6_ref);
        comp.featurematchings = (int)matches.size(); 
        comp.hashmatching = (double)diff6; 
        comp.image = rawname;
        comparison6.push_back(comp);
        hash_comparison6 << comp.image << " " << diff6 << " " << comp.featurematchings << endl;
      }
    }
    it++;
  }
  hashTableSize=count;
  // Close files
  h_out << hash_indexes.str();
  h1_out << hash_comparison1.str();
  h2_out << hash_comparison2.str();
  h3_out << hash_comparison3.str();
  h4_out << hash_comparison4.str();
  h5_out << hash_comparison5.str();
  h6_out << hash_comparison5.str();
  h_out.close();
  h1_out.close();
  h2_out.close();
  h3_out.close();
  h4_out.close();
  h5_out.close();
  h6_out.close();

  // Sort hashing vectors
  sort(comparison1.begin(), comparison1.end(), hash_matching::OpencvUtils::sorttrioByDistance);
  sort(comparison2.begin(), comparison2.end(), hash_matching::OpencvUtils::sorttrioByDistance);
  sort(comparison3.begin(), comparison3.end(), hash_matching::OpencvUtils::sorttrioByDistance);
  sort(comparison4.begin(), comparison4.end(), hash_matching::OpencvUtils::sorttrioByDistance);
  sort(comparison5.begin(), comparison5.end(), hash_matching::OpencvUtils::sorttrioByDistance);
  sort(comparison6.begin(), comparison6.end(), hash_matching::OpencvUtils::sorttrioByDistance);
 
  for(int i=0; i<(comparison1.size()); i++)
    ROS_INFO_STREAM("BEST MATCHING 1:   " << comparison1[i].image << " (" << comparison1[i].hashmatching << ")" << " (" << comparison1[i].featurematchings << ")");
  for(int i=0; i<(comparison2.size()); i++)
    ROS_INFO_STREAM("BEST MATCHING 2:   " << comparison2[i].image << " (" << comparison2[i].hashmatching << ")" << "(" << comparison2[i].featurematchings << ")");
  for(int i=0; i<(comparison3.size()); i++)
    ROS_INFO_STREAM("BEST MATCHING 3:   " << comparison3[i].image << " (" << comparison3[i].hashmatching << ")" << "(" << comparison3[i].featurematchings << ")");
  for(int i=0; i<(comparison4.size()); i++)
    ROS_INFO_STREAM("BEST MATCHING 4:   " << comparison4[i].image << " (" << comparison4[i].hashmatching << ")" << "(" << comparison4[i].featurematchings << ")");
  for(int i=0; i<(comparison5.size()); i++)
    ROS_INFO_STREAM("BEST MATCHING 5:   " << comparison5[i].image << " (" << comparison5[i].hashmatching << ")" << "(" << comparison5[i].featurematchings << ")");
  for(int i=0; i<(comparison6.size()); i++)
    ROS_INFO_STREAM("BEST MATCHING 6:   " << comparison6[i].image << " (" << comparison6[i].hashmatching << ")" << "(" << comparison6[i].featurematchings << ")");

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
  int size; 
  // ROS_ASSERT(hash_1.size() == hash_2.size());
  if (hash_1.size()< hash_2.size()) // lets take the minor of both sizes if they are not equal. 
    size=hash_1.size(); 
  else
    size=hash_2.size();

  double sum = 0.0;
  for (uint i=0; i<size; i++) // the module of the difference
  {
    if ((hash_1[i]!= NULL) && (hash_2[i]!= NULL)) 
    sum += pow((double)hash_1[i] - (double)hash_2[i], 2.0);
    //ROS_INFO_STREAM("hash_1;hash_2;" << (double)hash_1[i] << ";" << (double)hash_2[i] );
  }
  return sqrt(sum);
}

