#ifndef HASH_MATCHING_BASE_H
#define HASH_MATCHING_BASE_H

#include <ros/ros.h>
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

namespace hash_matching
{

class HashMatchingBase
{

public:
  // Constructor
  HashMatchingBase(ros::NodeHandle nh, ros::NodeHandle nhp);

  struct trio
  {
    float hash_matching;
    int feature_matchings;
    string image;

    // Struct constructors
    trio() : hash_matching(), feature_matchings(), image() {};
    trio(float h, int f, string i) : hash_matching(h), feature_matchings(f), image(i) {};
  };

protected:

  // Node handlers
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

private:
  float match(vector<float> hash_1, vector<float> hash_2);
  uint match(vector<uint> hash_1, vector<uint> hash_2);
  template <typename T> string toString( const T& n );
};

} // namespace

#endif // HASH_MATCHING_BASE_H