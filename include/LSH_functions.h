
#include "stereo_properties.h"
#include "opencv_utils.h"
#include <ros/ros.h>
#include <opencv2/core/eigen.hpp>

#define IntT int
#define LongUns64T long long unsigned
#define Uns32T unsigned
#define Int32T int
#define RealT long double

// 2^32-5
#define UH_PRIME_DEFAULT 4294967291U

// 2^29 = 536870912 = 5.36870912e+8
#define MAX_HASH_RND 536870912U
#define MIN_HASH_RND 1U
// #define MAX_HASH_RND 10U

// 2^32-1
#define TWO_TO_32_MINUS_1 4294967295U

Uns32T computeProductModDefaultPrime(vector<float> random_vector, vector<double> bucket, int size);
Uns32T computeUHashFunction(vector<float> random_vector, vector<double> bucket, IntT size, Uns32T prime, Int32T hashTableSize);
vector<float> compute_random_vector(uint seed, int size);