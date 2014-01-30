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
	double match(vector<double> hash_1, vector<double> hash_2);
	vector<double> computeHash(Hash hash_obj, Mat desc, vector<Hash>& hash_objs, hash_matching::Hash::Params hash_params);
};

} // namespace

#endif // HASH_MATCHING_BASE_H