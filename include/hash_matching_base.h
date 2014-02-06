#ifndef HASH_MATCHING_BASE_H
#define HASH_MATCHING_BASE_H

#include <ros/ros.h>
#include "opencv2/core/core.hpp"

// new for the new class Hash
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv_utils.h"
//#include <Eigen/Eigen>


using namespace std;
using namespace cv;

namespace hash_matching
{

// class Hash new class
// {
// public:

//   // Class contructor
//   Hash();

//   // Compute the SVD of the descriptors matrix
//   vector<float> computeSVD(Mat desc, int dim);
// };

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
