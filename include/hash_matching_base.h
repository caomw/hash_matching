#ifndef HASH_MATCHING_BASE_H
#define HASH_MATCHING_BASE_H

#include <ros/ros.h>
#include "opencv2/core/core.hpp"
#include "hash.h"

using namespace std;
using namespace cv;

namespace hash_matching
{

class HashMatchingBase
{

public:
	// Constructor
  HashMatchingBase(ros::NodeHandle nh, ros::NodeHandle nhp);

protected:

	// Node handlers
	ros::NodeHandle nh_;
	ros::NodeHandle nh_private_;

private:
	float match(vector<float> hash_1, vector<float> hash_2);
	vector<float> computeHash();
};

} // namespace

#endif // HASH_MATCHING_BASE_H