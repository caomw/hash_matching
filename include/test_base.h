#ifndef TEST_BASE_H
#define TEST_BASE_H

#include <ros/ros.h>
#include "opencv2/core/core.hpp"
#include "stereo_properties.h"

using namespace std;
using namespace cv;

namespace hash_matching
{

class TestBase
{

public:
  // Constructor
  TestBase(ros::NodeHandle nh, ros::NodeHandle nhp);

protected:

  // Node handlers
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

private:
  float match(vector<float> hash_1, vector<float> hash_2);
  template <typename T> string toString( const T& n );
  string getImageIdx(string filename);
  bool loopClosure(StereoProperties ref_prop,  
                   string cur_filename, 
                   double desc_thresh, 
                   int min_matches, 
                   int min_inliers,
                   int &matches,
                   int &inliers);
};

} // namespace

#endif // TEST_BASE_H