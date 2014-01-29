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
		double hashmatching;
		int featurematchings;
		string image;
		/* data */
	};
protected:

	// Node handlers
	ros::NodeHandle nh_;
	ros::NodeHandle nh_private_;

private:
	double match(vector<uint> hash_1, vector<uint> hash_2);
	double match(vector<double> hash_1, vector<double> hash_2);
	
};

} // namespace

#endif // HASH_MATCHING_BASE_H