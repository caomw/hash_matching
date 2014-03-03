#ifndef STEREO_PROPERTIES_H
#define STEREO_PROPERTIES_H

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv_utils.h"

using namespace std;
using namespace cv;

namespace hash_matching
{

class StereoProperties
{

public:

  // Class contructor
  StereoProperties();

  struct Params
  {
    //Default constructor sets all values to defaults.
    Params();

    // Class parameters
    string desc_type;               //!> Descriptor type can be: SIFT or SURF.
  };

  // Set the parameter struct
  void setParams(const Params& params);

  // Return current parameters
  inline Params params() const { return params_; }

  // Compute the keypoints and descriptors for the images
  void setImage(const Mat& img);

  // Return the image
  Mat getImg();

  // Return the keypoints of the image
  vector<KeyPoint> getKp();

  // Return the descriptors of the image
  Mat getDesc();

private:

  // Stores parameters
  Params params_;

  // Stereo vision properties
  Mat img_;                             //!> Stores the image
  vector<KeyPoint> kp_;                 //!> Unfiltered keypoints of the images.
  Mat desc_;                            //!> Unfiltered descriptors of the images.
};

} // namespace

#endif // STEREO_PROPERTIES_H