#ifndef HASH_H
#define HASH_H

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv_utils.h"

using namespace std;
using namespace cv;

namespace hash_matching
{

class Hash
{

public:

  // Class contructor
  Hash();

  struct Params
  {
    //Default constructor sets all values to defaults.
    Params();

    // Class parameters
    int num_hyperplanes;            //!> Number of hyperplanes to consider for hash.

    // Default values
    static const int            DEFAULT_NUM_HYPERPLANES = 4;
  };

  // Set the parameter struct
  void setParams(const Params& params);

  // Return current parameters
  inline Params params() const { return params_; }

  // Initialize class
  void initialize(Mat desc);

  // Compute the hash
  vector<double> computeHash(Mat desc, vector<KeyPoint> kp);

  // Access specifiers
  void showHash(vector<double> hash);

private:

  // Save the hyperplanes into a file for plotting
  void saveToFile(Mat desc);

  // Creates the combination table for hash
  void createCombinations();

  // Recursive function to create the combinations table for hash
  void recursiveCombinations(const vector< vector<string> > &all_vecs, 
                             size_t vec_idx, vector<string> combinations,
                             vector< vector<string> > &result);

  // Compute the hyperplanes
  void computeHyperplanes(Mat desc,
                          vector<float>& centroid,
                          vector< vector<float> >& H, 
                          vector<float>& delta);

  // Compute the hash measure for a set of descriptors
  vector<double> hashMeasure(Mat desc, 
                             vector<KeyPoint> kp,
                             vector<float> centroid, 
                             vector< vector<float> > H, 
                             vector<float> delta);

  vector< vector<int> > computeRegions(Mat desc,
                                       vector< vector<float> > H, 
                                       vector<float> delta);

  double angleBetweenVectors(vector<float> vector_a, 
                             vector<float> vector_b);

  double moduleBetweenVectors(vector<float> vector_a, 
                              vector<float> vector_b);

  // Stores parameters
  Params params_;

  // Stereo vision properties
  vector<string> comb_;                     //!> Table of possible hash combinations
  vector<float> centroid_;                  //!> Save the main centroid
  vector< vector<float> > H_;               //!> Save the main H
  vector<float> delta_;                     //!> Save the main delta
  vector< vector<float> > sub_centroid_;    //!> Save the sub-region centroids
  vector< vector< vector<float> > > sub_H_; //!> Save the sub-region H
  vector< vector<float> > sub_delta_;       //!> Save the sub-region deltas
};

} // namespace

#endif // HASH_H