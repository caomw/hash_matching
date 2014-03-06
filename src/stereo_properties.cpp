#include "stereo_properties.h"
#include "opencv_utils.h"
#include <ros/ros.h>

/** \brief Parameter constructor. Sets the parameter struct to default values.
  */
hash_matching::StereoProperties::Params::Params() :
  desc_type("SIFT")
{}

/** \brief StereoProperties constructor
  */
hash_matching::StereoProperties::StereoProperties() {}

/** \brief Sets the parameters
  * \param parameter struct.
  */
void hash_matching::StereoProperties::setParams(const Params& params) 
{
  params_ = params;
}

// Access specifiers
Mat hash_matching::StereoProperties::getImg() { return img_; }
vector<KeyPoint> hash_matching::StereoProperties::getKp() { return kp_; }
Mat hash_matching::StereoProperties::getDesc() { return desc_; }


void hash_matching::StereoProperties::setImage(const Mat& img)
{
  img_ = img;

  // Extract keypoints
  desc_ = Mat_< vector<float> >();
  hash_matching::OpencvUtils::keypointDetector(img_, kp_, params_.desc_type);

  // Extract descriptors
  hash_matching::OpencvUtils::descriptorExtraction(img_, kp_, desc_, params_.desc_type);
}