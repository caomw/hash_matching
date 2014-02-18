#include "hash.h"
#include "opencv_utils.h"
#include <ros/ros.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/core/eigen.hpp>

// Parameter constructor. Sets the parameter struct to default values.
hash_matching::Hash::Params::Params() :
  num_hyperplanes(DEFAULT_NUM_HYPERPLANES)
{}

// Hash constructor
hash_matching::Hash::Hash() {}

// Sets the parameters
void hash_matching::Hash::setParams(const Params& params) 
{
  params_ = params;
}

bool hash_matching::Hash::initialize(Mat desc)
{
  // Try to build the hyperplanes
  int counter = 0;
  bool init_done = false;
  while(!init_done && counter<10)
  {
    init_done = initializeHyperplanes(desc);
    counter++;
  }
  return init_done;
}

bool hash_matching::Hash::initializeHyperplanes(Mat desc)
{
  // Initialize class properties
  H_.clear();
  delta_.clear();
  sub_H_.clear();
  sub_delta_.clear();

  // Compute combinations
  createCombinations();

  // Compute main hyperplanes
  vector< vector<float> > H;
  vector<float> delta;
  computeHyperplanes(desc, H, delta);
  H_ = H;
  delta_ = delta;

  // Compute the regions
  vector< vector<int> > hash_idx = computeRegions(desc, H_, delta_);

  // Compute the hyperplanes for every region
  for (uint i=0; i<comb_.size(); i++)
  {
    // Get the descriptors for this subregion
    vector<int> indices = hash_idx[i];
    Mat region_desc;
    for (uint n=0; n<indices.size(); n++)
      region_desc.push_back(desc.row(indices[n]));

    // Detect empty regions
    if(region_desc.rows == 0)
      return false;

    // Compute the hyperplanes for this region
    H.clear();
    delta.clear();
    computeHyperplanes(region_desc, H, delta);
    sub_H_.push_back(H);
    sub_delta_.push_back(delta);
    region_desc.release();
  }
  return true;
}

// Creates the combinations table
void hash_matching::Hash::createCombinations()
{
  vector<string> h_v;
  h_v.push_back("0");
  h_v.push_back("1");

  int d = params_.num_hyperplanes;
  vector< vector<string> > comb_table;
  for (int i=0; i<d; i++)
    comb_table.push_back(h_v);

  vector<string> combinations;
  vector< vector<string> > result;
  recursiveCombinations(comb_table, 0, combinations, result);

  comb_.clear();
  for (uint i=0; i<result.size(); i++)
  {
    string t = "";
    for (uint j=0; j<result[i].size(); j++)
    {
      t += result[i][j];
    }
    comb_.push_back(t);
  }
}

// Recursive function to create the table of possible combinations
void hash_matching::Hash::recursiveCombinations(const vector< vector<string> > &all_vecs, 
                                                      size_t vec_idx, vector<string> combinations,
                                                      vector< vector<string> > &result)
{
  if (vec_idx >= all_vecs.size())
  {
    result.push_back(combinations);
    return;
  }
  
  for (size_t i=0; i<all_vecs[vec_idx].size(); i++)
  {
    vector<string> tmp = combinations;
    tmp.push_back(all_vecs[vec_idx][i]);
    recursiveCombinations(all_vecs, vec_idx+1, tmp, result);
  }
}

// Computes the hyperplanes
void hash_matching::Hash::computeHyperplanes(Mat desc,
                                             vector< vector<float> >& H, 
                                             vector<float>& delta)
{
  // Set the number of hyperplanes
  int d = params_.num_hyperplanes;

  // Compute the descriptors centroid
  vector<float> centroid;
  centroid.clear();
  for (int n=0; n<desc.cols; n++)
  {
    float sum = 0.0;
    for (int m=0; m<desc.rows; m++)
    {
      float val = desc.at<float>(m, n);
      sum += val;
    }
    centroid.push_back(sum / desc.rows);
  }

  // Generate 'd' random hyperplanes
  // srand(101);
  H.clear();
  for (int i=0; i<d; i++)
  {
    vector<float> h;
    for(int n=0; n<desc.cols; n++)
    {
      float val = ((float(rand()) / float(RAND_MAX)) * (1 + 1)) -1.0;
      h.push_back(val);
    }
    H.push_back(h);
  }

  // Make hyperplanes pass through centroid
  delta.clear();
  for (int i=0; i<d; i++)
  {
    float f = 0.0;
    for(int n=0; n<desc.cols; n++)
    {
      f -= (float)H[i][n] * (float)centroid[n];
    }
    delta.push_back(f);
  }
}

vector<double> hash_matching::Hash::computeHash(Mat desc)
{
  // Initialize hash
  vector<double> hash;

  // Compute the regions
  vector< vector<int> > hash_idx = computeRegions(desc, H_, delta_);

  // Iterate over major retions
  for (uint i=0; i<hash_idx.size(); i++)
  {
    // Get the descriptors for this region
    vector<int> indices = hash_idx[i];
    Mat region_desc;
    for (uint n=0; n<indices.size(); n++)
      region_desc.push_back(desc.row(indices[n]));

    vector< vector<int> > sub_hash_idx = computeRegions(region_desc, sub_H_[i], sub_delta_[i]);

    // Iterate over sub-regions
    for (uint j=0; j<sub_hash_idx.size(); j++)
    {
      // Get the descriptors for this subregion
      vector<int> sub_indices = sub_hash_idx[j];
      Mat sub_region_desc;
      for (uint n=0; n<sub_indices.size(); n++)
        sub_region_desc.push_back(region_desc.row(sub_indices[n]));

      // Compute the hash for every subregion and attach to the main hash
      vector<double> sub_hash = hashMeasure(sub_region_desc);
      for (uint k=0; k<sub_hash.size(); k++)
        hash.push_back(sub_hash[k]);

      // Clear
      sub_region_desc.release();
    }
    // Clear
    region_desc.release();
  }
  return hash;
}

vector<double> hash_matching::Hash::hashMeasure(Mat desc)
{
  // Sanity check
  vector<double> hash(3, 0);
  if (desc.rows < 3)
    return hash;

  // Convert the cv Mat to Eigen::Matrix3f
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> h;
  cv2eigen(desc, h);

  // Compute the SVD
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(h, Eigen::ComputeThinU | Eigen::ComputeThinV);

  // Get the singular vectors for the main singular value
  Eigen::Vector3f sv = svd.singularValues();
  Eigen::Vector3f u = svd.matrixU().col(0);
  Eigen::Vector3f v = svd.matrixV().col(0);

  // Sanity check
  if (sv.size() != 3)
    return hash;

  // Mount
  hash.clear();
  for (uint i=0; i<3; i++)
    hash.push_back((double)u[i]*sv[i]);
  
  return hash;
}

vector< vector<int> > hash_matching::Hash::computeRegions(Mat desc,
                                                          vector< vector<float> > H, 
                                                          vector<float> delta)
{
  // Initialize the hash indices table
  vector< vector<int> > hash_idx;
  for (uint i=0; i<comb_.size(); i++)
  {
    vector<int> t;
    hash_idx.push_back(t);
  }

  // Sanity check
  if (H.size() < 1) {
    ROS_ERROR("[Hash:] No hyperplanes received!");
    return hash_idx;
  }
  if (H[0].size() < 1) {
    ROS_ERROR("[Hash:] At least, one hyperplane is empty!");
    return hash_idx;
  }

  // Set the number of hyperplanes
  int d = params_.num_hyperplanes;

  // Compute the hash
  for(int i=0; i<desc.rows; i++)
  {
    string bin = "";
    for (int n=0; n<d; n++)
    {
      float v = 0.0;
      for(int m=0; m<desc.cols; m++)
        v += (float)H[n][m] * desc.at<float>(i, m);
      v += delta[n];

      if(v>0)
        bin += "1";
      else
        bin += "0";
    }

    // Get the position of this bin and increase hash
    int pos = find(comb_.begin(), comb_.end(), bin) - comb_.begin();

    // Update hash indices table
    vector<int> t;
    if (hash_idx[pos].size() != 0)
      t = hash_idx[pos];
    t.push_back(i);
    hash_idx[pos] = t;
  }
  return hash_idx;
}