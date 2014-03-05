#ifndef HASH_H
#define HASH_H

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv_utils.h"
#include <Eigen/Eigen>

using namespace std;
using namespace cv;

namespace hash_matching
{

class Hash
{

public:

  // Class contructor
  Hash();

  // Initialize class
  bool initialize(Mat desc, int max_features);
  bool initializeHyperplanes(Mat desc, int max_features, int &region_size);

  // Compute the hash
  vector<float> computeHash(Mat desc);

private:

  // Save the hyperplanes into a file for plotting
  void saveToFile(Mat desc, string sub);

  // Creates the combination table for hash
  void createCombinations();

  // Recursive function to create the combinations table for hash
  void recursiveCombinations(const vector< vector<string> > &all_vecs, 
                             size_t vec_idx, vector<string> combinations,
                             vector< vector<string> > &result);

  // Compute the hyperplanes
  void computeHyperplanes(Mat desc,
                          vector< vector<float> >& H, 
                          vector<float>& delta,
                          vector<float> centroid);

  // Compute the hash measure for a set of descriptors
  vector <float> hashMeasure(Mat desc, int seed);

  // Divide descriptors by region
  vector< vector<int> > computeRegions(Mat desc,
                                       vector< vector<float> > H, 
                                       vector<float> delta);

  // Compute a random vector
  vector<float> compute_random_vector(uint seed, int size);

  // Stereo vision properties
  vector<string> comb_;                     //!> Table of possible hash combinations
  vector< vector<float> > H_;               //!> Save the main H
  vector<float> delta_;                     //!> Save the main delta
  vector<float> centroid_;                  //!> Save the main centroid
  vector< vector< vector<float> > > sub_H_; //!> Save the sub-region H
  vector< vector<float> > sub_delta_;       //!> Save the sub-region deltas  
  vector< vector<float> > sub_centroid_;    //!> Save the sub-region centroid
  vector< vector<int> > sub_seeds_;         //!> Save the sub-region seeds
  int num_hyperplanes_;                     //!> Number of hyperplanes
};

} // namespace

#endif // HASH_H