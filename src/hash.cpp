#include "hash.h"
#include "utils.h"
#include <ros/ros.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/core/eigen.hpp>


/** \brief Parameter constructor. Sets the parameter struct to default values.
  */
hash_matching::Hash::Params::Params() :
  proj_num(DEFAULT_PROJ_NUM),
  features_max_value(DEFAULT_F_MAX_VALUE),
  features_min_value(DEFAULT_F_MIN_VALUE),
  n_levels(DEFAULT_n_levels)
{}

// Hash constructor
hash_matching::Hash::Hash()
{
  // Initializations
  num_hyperplanes_ = -1;
  h1_size_ = -1;
  h2_size_ = -1;
  h3_size_ = -1;
}

// Sets the parameters
void hash_matching::Hash::setParams(const Params& params) 
{
  params_ = params;
}

// Get the number of hyperplanes
int hash_matching::Hash::getHyperplanes(){return num_hyperplanes_;}

// Class initialization
bool hash_matching::Hash::init(Mat desc, bool proj_orthogonal)
{
  // Create the random projections vectors 
  initProjections(desc.rows, proj_orthogonal);

  // For hash histogram
  q_interval_ = (params_.features_max_value - params_.features_min_value)/(float)params_.n_levels;

  // Setup the size of hashes. The size of hash 1 is computed in the initialization
  h1_size_ = 1;
  h2_size_ = params_.n_levels;
  h3_size_ = params_.proj_num * desc.cols;

  // Get the descriptors type: hyperplanes can only be generated with descriptors of type 32FC1
  string type = hash_matching::Utils::matType2str(desc.type());
  if (type != "32FC1") return true;

  // Try to build the hyperplanes
  int max_hyperplanes = 20;
  bool init_done = false;
  num_hyperplanes_ = 15;
  while(!init_done && num_hyperplanes_>1)
  {
    int region_size;
    init_done = initHyperplanes(desc, region_size);
    ROS_INFO_STREAM("[Hash:] Initializing iteration with " << num_hyperplanes_ << " hyperplanes (" << region_size << "): " << init_done);
    num_hyperplanes_--;
  }
  num_hyperplanes_++;

  // Log
  if (!init_done)
    ROS_ERROR("[Hash:] Impossible to find a correct number of hyperplanes!");
  else
    ROS_INFO_STREAM("[Hash:] Initialization finishes with " << num_hyperplanes_ << " hyperplanes.");

  return init_done;
}

// Hyperplanes initialization
bool hash_matching::Hash::initHyperplanes(Mat desc, int &region_size)
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
  vector< vector<uint> > reg_idx = computeRegions(desc, H_, delta_);

  // Setup the size of hash 1
  h1_size_ = int(pow(comb_.size(), 2.0));

  // Compute the hyperplanes for every region
  region_size = -1;
  int seed = time(NULL);
  for (uint i=0; i<comb_.size(); i++)
  {
    // Get the descriptors for this subregion
    vector<uint> indices = reg_idx[i];
    Mat region_desc;
    for (uint n=0; n<indices.size(); n++)
      region_desc.push_back(desc.row(indices[n]));

    // Detect regions with no descriptors
    if(region_desc.rows <= 0)
    {
      region_size = region_desc.rows;
      return false;
    }

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

  int d = num_hyperplanes_;
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
  int d = num_hyperplanes_;

  // Compute the descriptors centroid
  vector<float> centroid;
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

vector< vector<uint> > hash_matching::Hash::computeRegions(Mat desc,
                                                          vector< vector<float> > H, 
                                                          vector<float> delta)
{
  // Initialize the hash indices table
  vector< vector<uint> > reg_idx;
  for (uint i=0; i<comb_.size(); i++)
  {
    vector<uint> t;
    reg_idx.push_back(t);
  }

  // Sanity check
  if (H.size() < 1) {
    ROS_ERROR("[Hash:] No hyperplanes received!");
    return reg_idx;
  }
  if (H[0].size() < 1) {
    ROS_ERROR("[Hash:] At least, one hyperplane is empty!");
    return reg_idx;
  }

  // Compute the hash
  for(int i=0; i<desc.rows; i++)
  {
    string bin = "";
    for (int n=0; n<num_hyperplanes_; n++)
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
    vector<uint> t;
    if (reg_idx[pos].size() != 0)
      t = reg_idx[pos];
    t.push_back((uint)i);
    reg_idx[pos] = t;

  }

  return reg_idx;
}

// Compute the hash of the hyperplanes
vector<uint> hash_matching::Hash::getHash1(Mat desc)
{
  // Initialize the hash with 0's
  vector<uint> hash(h1_size_, 0);

  // Get the descriptors type: hyperplanes can only be generated with descriptors of type 32FC1
  string type = hash_matching::Utils::matType2str(desc.type());
  if (type != "32FC1" || desc.rows == 0) return hash;

  // Compute the regions
  vector< vector<uint> > reg_idx = computeRegions(desc, H_, delta_);

  // Iterate over major retions
  uint n = 0;
  for (uint i=0; i<reg_idx.size(); i++)
  {
    // Get the descriptors for this region
    vector<uint> indices = reg_idx[i];
    Mat region_desc;
    for (uint j=0; j<indices.size(); j++)
      region_desc.push_back(desc.row(indices[j]));

    // Compute the sub-regions
    vector< vector<uint> > sub_reg_idx = computeRegions(region_desc, sub_H_[i], sub_delta_[i]);

    // Iterate over sub-regions
    for (uint j=0; j<sub_reg_idx.size(); j++)
    {
      // Count the descriptors for this subregion
      hash[n] = (uint)sub_reg_idx[j].size();
      n++;
    }
    // Clear
    region_desc.release();
  }

  return hash;
}

// Compute the hash of the feature quantization histogram
vector<uint> hash_matching::Hash::getHash2(Mat desc)
{
  // Initialize the hash with 0's
  vector<uint> hash(h2_size_, 0);

  // Get the descriptors type: the feature histogram can only be generated with descriptors of type 32FC1
  string type = hash_matching::Utils::matType2str(desc.type());
  if (type != "32FC1" || desc.rows == 0) return hash;

  for(int m=0; m<desc.rows; m++)
  {
    for(int n=0; n<desc.cols; n++)
    {
      float resto = fmodf(desc.at<float>(m, n), q_interval_);
      int integer_part = (int)( ( (desc.at<float>(m, n) - params_.features_min_value) / q_interval_ ) + 0.5 );

      if (resto > 0.0)
        hash[integer_part+1]++;
      else 
        hash[integer_part]++;
    }
  }
  return hash;
}

// Compute the hash of descriptor projections
vector<float> hash_matching::Hash::getHash3(Mat desc)
{
  // initialize the histogram with 0's
  vector<float> hash(h3_size_, 0.0);

  // Sanity check
  if (desc.rows == 0) return hash;

  // Convert descriptors if needed
  string type = hash_matching::Utils::matType2str(desc.type());
  if (type != "32FC1") desc.convertTo(desc, CV_32F);

  // Project the descriptors
  uint k = 0;
  for (uint i=0; i<r_.size(); i++)
  {
    for (int n=0; n<desc.cols; n++)
    {
      float desc_sum = 0.0;
      for (uint m=0; m<desc.rows; m++)
      {
        desc_sum += r_[i][m]*desc.at<float>(m, n);
      }
      hash[k] = desc_sum;
      k++;
    }
  }

  // Normalize hash
  return normalize_vector(hash);
}

void hash_matching::Hash::initProjections(int desc_size, bool orthogonal)
{
  // Initializations
  int seed = time(NULL);
  r_.clear();

  // The size of the descriptors may vary... We multiply the current descriptor size 
  // for a escalar to handle the larger cases.
  int v_size = 6*desc_size;

  if (orthogonal)
  {
    // We will generate N-orthogonal vectors creating a linear system of type Ax=b.

    // Generate a first random vector
    vector<float> r = compute_random_vector(seed, v_size);
    r_.push_back(unit_vector(r));

    // Generate the set of orthogonal vectors
    for (uint i=1; i<params_.proj_num; i++)
    {
      // Generate a random vector of the correct size
      vector<float> new_v = compute_random_vector(seed + i, v_size - i);

      // Get the right terms (b)
      VectorXf b(r_.size());
      for (uint n=0; n<r_.size(); n++)
      {
        vector<float> cur_v = r_[n];
        float sum = 0.0;
        for (uint m=0; m<new_v.size(); m++)
        {
          sum += new_v[m]*cur_v[m];
        }
        b(n) = -sum;
      }

      // Get the matrix of equations (A)
      MatrixXf A(i, i);
      for (uint n=0; n<r_.size(); n++)
      {
        uint k=0;
        for (uint m=r_[n].size()-i; m<r_[n].size(); m++)
        {
          A(n,k) = r_[n][m];
          k++;
        }
      }

      // Apply the solver
      VectorXf x = A.colPivHouseholderQr().solve(b);

      // Add the solutions to the new vector
      for (uint n=0; n<r_.size(); n++)
        new_v.push_back(x(n));

      // Push the new vector
      r_.push_back(unit_vector(new_v));
    }
  }
  else
  {
    for (uint i=0; i<params_.proj_num; i++)
    {
      vector<float> r = compute_random_vector(seed + i, v_size);
      r_.push_back(unit_vector(r));
    }
  }
  
  
}

// Computes a random vector of some size
vector<float> hash_matching::Hash::compute_random_vector(uint seed, int size)
{
  srand(seed);
  vector<float> h;
  for (int i=0; i<size; i++)
    h.push_back( (float)rand()/RAND_MAX );
  return h;
}

// Make a vector unit
vector<float> hash_matching::Hash::unit_vector(vector<float> x)
{
  // Compute the norm
  float sum = 0.0;
  for (uint i=0; i<x.size(); i++)
    sum += pow(x[i], 2.0);
  float x_norm = sqrt(sum);

  // x^ = x/|x|
  for (uint i=0; i<x.size(); i++)
    x[i] = x[i] / x_norm;

  return x;
}

// Normalize vector
vector<float> hash_matching::Hash::normalize_vector(vector<float> x)
{
  // Get the maximum and minimum values
  float min = *min_element(x.begin(), x.end());
  float max = *max_element(x.begin(), x.end());

  // Normalize
  for (uint i=0; i<x.size(); i++)
    x[i] = (x[i]-min)/(max-min);

  return x;
}