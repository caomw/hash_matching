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
  bool initialize(Mat desc, int proj_num);

  // Compute the hash
  vector<float> computeHash(Mat desc);

private:

  // Compute a random vector
  vector<float> compute_random_vector(uint seed, int size);

  // Properties
  vector< vector<float> > r_;           //!> Vector of random values
};

} // namespace

#endif // HASH_H