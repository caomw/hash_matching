#include "hash.h"
#include "opencv_utils.h"
#include <ros/ros.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/core/eigen.hpp>

// Hash constructor
hash_matching::Hash::Hash() {}

bool hash_matching::Hash::initialize(Mat desc, int proj_num)
{
  // Sanity check
  if (proj_num <= 0)
  {
    ROS_ERROR("[Hash:] Number of projections must be a positive integer.");
    return false;
  }

  // Compute the random vectors
  int seed = time(NULL);
  for (uint i=0; i<proj_num; i++)
  {
    vector<float> r = compute_random_vector(seed + i, 3*desc.rows);
    r_.push_back(r);
  }
  return true;  
}

vector<float> hash_matching::Hash::computeHash(Mat desc)
{
  // Project the descriptors
  vector<float> h;
  for (uint i=0; i<r_.size(); i++)
  {
    for (int n=0; n<desc.cols; n++)
    {
      float desc_sum = 0.0;
      for (uint m=0; m<desc.rows; m++)
      {
        desc_sum += r_[i][m]*desc.at<float>(m, n);
      }
      h.push_back(desc_sum);
    }
  }
  return h;
}

vector<float> hash_matching::Hash::compute_random_vector(uint seed, int size)
{
  srand(seed);
  vector<float> h;
  for (int i=0; i<size; i++)
    h.push_back( ((float) rand() / (RAND_MAX)) );
  return h;
}