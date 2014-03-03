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
  bool initialize(Mat desc);
  bool initializeHyperplanes(Mat desc, int &region_size);

  // Compute the hash
  vector<double> computeHash(Mat desc);

  // Access specifiers
  void showHash(vector<double> hash);

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
                          vector<float>& delta);

  // Compute the hash measure for a set of descriptors
  double hashMeasure(Mat desc);

  vector< vector<int> > computeRegions(Mat desc,
                                       vector< vector<float> > H, 
                                       vector<float> delta);

  // Stereo vision properties
  vector<string> comb_;                     //!> Table of possible hash combinations
  vector< vector<float> > H_;               //!> Save the main H
  vector<float> delta_;                     //!> Save the main delta
  vector< vector< vector<float> > > sub_H_; //!> Save the sub-region H
  vector< vector<float> > sub_delta_;       //!> Save the sub-region deltas
  int num_hyperplanes_;                     //!> Number of hyperplanes
};

} // namespace

#endif // HASH_H