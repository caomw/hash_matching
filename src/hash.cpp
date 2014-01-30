#include "hash.h"
#include "opencv_utils.h"
#include <ros/ros.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>

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

void hash_matching::Hash::initialize(Mat desc)
{
  // Save the hyperplanes to file
  saveToFile(desc);

  // Initialize class properties
  centroid_.clear();
  H_.clear();
  delta_.clear();
  sub_centroid_.clear();
  sub_H_.clear();
  sub_delta_.clear();

  // Compute combinations
  createCombinations();

  // Compute main hyperplanes
  vector<float> centroid;
  vector< vector<float> > H;
  vector<float> delta;
  computeHyperplanes(desc, centroid, H, delta);
  centroid_ = centroid;
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

    // Compute the hyperplanes for this region
    centroid.clear();
    H.clear();
    delta.clear();
    computeHyperplanes(region_desc, centroid, H, delta);
    sub_centroid_.push_back(centroid);
    sub_H_.push_back(H);
    sub_delta_.push_back(delta);
    region_desc.release();
  }
}

// Saves the hyperplanes and descriptors to file for plotting in 3D
void hash_matching::Hash::saveToFile(Mat desc)
{
  // Reduce the descriptors to 3 dimensions
  Mat sub_desc = desc(cv::Rect(0,0,3,desc.rows));

  // Compute the hyperplanes for this subset of descriptors
  vector<float> centroid;
  vector< vector<float> > H;
  vector<float> delta;
  computeHyperplanes(sub_desc, centroid, H, delta);

  // Write to file
  string hyperplanes = "/home/plnegre/Workspace/ROS_hydro/src/sandbox/hash_matching/data/hyperplanes.txt";
  string descriptors = "/home/plnegre/Workspace/ROS_hydro/src/sandbox/hash_matching/data/descriptors.txt";
  fstream f_hp(hyperplanes.c_str(), ios::out | ios::trunc);
  fstream f_desc(descriptors.c_str(), ios::out | ios::trunc);
  f_hp << setprecision(4) << 
          centroid[0] << "," << 
          centroid[1] << "," << 
          centroid[2] << "," <<
          0.0 << endl;
  for (uint i=0; i<H.size(); i++)
  {
    // Calculate two points of the hyperplane
    vector<float> h = H[i];
    f_hp << setprecision(4) << 
            h[0]  << "," << 
            h[1]  << "," << 
            h[2]  << "," << 
            delta[i]  << endl;
  }
  for (int i=0; i<sub_desc.rows; i++)
  {
    f_desc << setprecision(4) << 
              sub_desc.at<float>(i, 0) << "," << 
              sub_desc.at<float>(i, 1) << "," << 
              sub_desc.at<float>(i, 2) << endl;
  }
  f_hp.close();
  f_desc.close();
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
                                             vector<float>& centroid,
                                             vector< vector<float> >& H, 
                                             vector<float>& delta)
{
  // Set the number of hyperplanes
  int d = params_.num_hyperplanes;

  // Compute the descriptors centroid
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
  srand(101);
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

vector<double> hash_matching::Hash::computeHash(Mat desc, vector<KeyPoint> kp)
{
  // Initialize hash
  vector<double> hash;

  // Compute the regions
  vector< vector<int> > hash_idx = computeRegions(desc, H_, delta_);

  // Compute the hash measure for every region
  for (uint i=0; i<comb_.size(); i++)
  {
    // Get the descriptors for this subregion
    vector<int> indices = hash_idx[i];
    Mat region_desc;
    vector<KeyPoint> region_kp;
    for (uint n=0; n<indices.size(); n++)
    {
      region_desc.push_back(desc.row(indices[n]));
      region_kp.push_back(kp[indices[n]]);
    }

    // Compute the hash measure for this region
    vector<double> sub_hash = hashMeasure(region_desc, region_kp, sub_centroid_[i], sub_H_[i], sub_delta_[i]);
    region_desc.release();

    for (uint j=0; j<sub_hash.size(); j++)
      hash.push_back(sub_hash[j]);
  }
  return hash;
}

vector<double> hash_matching::Hash::hashMeasure(Mat desc,
                                                vector<KeyPoint> kp, 
                                                vector<float> centroid, 
                                                vector< vector<float> > H, 
                                                vector<float> delta)
{
  // Sanity check
  vector<double> hash(comb_.size(), 0);
  if (H.size() < 1)
    return hash;
  if (H[0].size() < 1)
    return hash;

  // Compute the regions for this set of descriptors
  vector< vector<int> > hash_idx = computeRegions(desc, H, delta);
  for (uint i=0; i<hash_idx.size(); i++)
  {
    // Get the indices for this region
    vector<int> indices = hash_idx[i];

    if (indices.size()==0)
    {
      hash.push_back(0.0);
      continue;
    }

    // Get the best kp for this region
    double best_response = 0.0;
    int best_idx = -1;
    for (uint n=0; n<indices.size(); n++)
    {
      if (kp[indices[n]].size > best_response)
      {
        best_response = (double)kp[indices[n]].size;
        best_idx = indices[n];
      }
    }

    vector<float> p_i;
    for (int n=0; n<desc.cols; n++)
      p_i.push_back(desc.at<float>(best_idx, n));

    // Compute the centroid for this region
    vector<float> region_centroid;
    for (int n=0; n<desc.cols; n++)
    {
      float mean = 0.0;
      for (uint m=0; m<indices.size(); m++)
      {
        mean += desc.at<float>(indices[m], n);
      }
      mean /= indices.size();
      region_centroid.push_back(mean);
    }

    // Module between vectors
    double module = moduleBetweenVectors(region_centroid, p_i);

    hash.push_back(best_response);
  }

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
  if (H.size() < 1)
    return hash_idx;
  if (H[0].size() < 1)
    return hash_idx;

  // Set the number of hyperplanes
  int d = params_.num_hyperplanes;

  // Compute the hash
  for(int i=0; i<desc.rows; i++)
  {
    string bin = "";
    float desc_mean = 0.0;
    for (int n=0; n<d; n++)
    {
      float v = 0.0;
      for(int m=0; m<desc.cols; m++)
      {
        v += (float)H_[n][m] * desc.at<float>(i, m);
        
        // Compute descriptor mean
        if(n == d-1)
          desc_mean += desc.at<float>(i, m);
      }
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

// Compute the angle between two vectors
double hash_matching::Hash::angleBetweenVectors(vector<float> vector_a, 
                                                vector<float> vector_b)
{
  double num = 0.0;
  for (uint j=0; j<vector_a.size(); j++)
  {
    num += (double)vector_a[j]*(double)vector_b[j];
  }

  // Compute the denominator of the phase
  double mod_a = 0.0;
  double mod_b = 0.0;
  for (uint j=0; j<vector_a.size(); j++)
  {
    mod_a += pow((double)vector_a[j], 2);
    mod_b += pow((double)vector_b[j], 2);
  }
  double den = sqrt(mod_a) * sqrt(mod_b);

  // Compute the phase
  double angle = (double)acos(num/den);

  if (!isfinite(angle))
    angle = 0.0;

  return angle;
}

// Compute the module between two vectors
double hash_matching::Hash::moduleBetweenVectors(vector<float> vector_a, 
                                                 vector<float> vector_b)
{
  double mod = 0.0;
  for (uint j=0; j<vector_a.size(); j++)
  {
    mod += pow((double)vector_a[j]-(double)vector_b[j], 2);
  }

  return sqrt(mod);
}