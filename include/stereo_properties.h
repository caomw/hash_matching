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
    int num_hyperplanes;            //!> Number of hyperplanes to consider for hash.
    int bucket_width;               //!> Bucket width.
    int bucket_height;              //!> Bucket height.
    int bucket_max;                 //!> Maximum number the features per bucket.

    // Default values
    static const int            DEFAULT_NUM_HYPERPLANES = 4;
    static const int            DEFAULT_BUCKET_WIDTH = 30;
    static const int            DEFAULT_BUCKET_HEIGHT = 30;
    static const int            DEFAULT_BUCKET_MAX = 10;
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

  vector<float> getCentroid();
  vector< vector<float> > getH();
  vector<float> getDelta();
  void computeHyperplanes();
  void setHyperplanes(vector<float> centroid, vector< vector<float> > H, vector<float> delta);

  // Return the hash of the left image
  vector<uint> getHash1();
  vector<double> getHash2();
  vector<double> getHash3();

  // Computes the hash for the left image
  void computeHash();

private:

  // Creates the combination table for hash
  void createCombinations();

  // Recursive function to create the combinations table for hash
  void recursiveCombinations(const vector< vector<string> > &all_vecs, 
                             size_t vec_idx, vector<string> combinations,
                             vector< vector<string> > &result);

  // Stores parameters
  Params params_;

  // Stereo vision properties
  Mat img_;                             //!> Stores the image
  vector<KeyPoint> kp_;                 //!> Unfiltered keypoints of the images.
  Mat desc_;                            //!> Unfiltered descriptors of the images.
  vector<uint> hash1_;                  //!> Hash vector of the image
  vector<double> hash2_;                //!> Hash vector of the image
  vector<double> hash3_;                //!> Hash vector of the image
  vector<string> comb_;                 //!> Table of possible hash combinations
  vector<float> centroid_;              //!> Saves the descriptor centroid  
  vector< vector<float> > H_;           //!> Hash hyperplanes
  vector<float> delta_;                 //!> Hyperplane variable
};

} // namespace

#endif // STEREO_PROPERTIES_H