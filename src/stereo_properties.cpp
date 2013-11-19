#include "stereo_properties.h"
#include "opencv_utils.h"
#include <ros/ros.h>

/** \brief Parameter constructor. Sets the parameter struct to default values.
  */
hash_matching::StereoProperties::Params::Params() :
  desc_type("SIFT"),
  num_hyperplanes(DEFAULT_NUM_HYPERPLANES),
  bucket_width(DEFAULT_BUCKET_WIDTH),
  bucket_height(DEFAULT_BUCKET_HEIGHT),
  bucket_max(DEFAULT_BUCKET_MAX)
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

  // Recompute combinations
  createCombinations();
}

// Access specifiers
Mat hash_matching::StereoProperties::getImg() { return img_; }
vector<KeyPoint> hash_matching::StereoProperties::getKp() { return kp_; }
Mat hash_matching::StereoProperties::getDesc() { return desc_; }
vector<uint> hash_matching::StereoProperties::getHash1() { return hash1_; }
vector<double> hash_matching::StereoProperties::getHash2() { return hash2_; }
vector<double> hash_matching::StereoProperties::getHash3() { return hash3_; }
vector<float> hash_matching::StereoProperties::getCentroid() { return centroid_; }
vector< vector<float> > hash_matching::StereoProperties::getH() { return H_; }
vector<float> hash_matching::StereoProperties::getDelta() { return delta_; }
void hash_matching::StereoProperties::setHyperplanes(vector<float> centroid, vector< vector<float> > H, vector<float> delta)
{
  centroid_ = centroid;
  H_ = H;
  delta_ = delta;
}


void hash_matching::StereoProperties::setImage(const Mat& img)
{
  img_ = img;

  // Extract keypoints and descriptors of reference image
  desc_ = Mat_< vector<float> >();
  hash_matching::OpencvUtils::keypointDetector(img_, kp_, params_.desc_type);
  
  // Bucket keypoints
  kp_ = hash_matching::OpencvUtils::bucketKeypoints(kp_, 
                                                    params_.bucket_width, 
                                                    params_.bucket_height, 
                                                    params_.bucket_max);
  hash_matching::OpencvUtils::descriptorExtraction(img_, kp_, desc_, params_.desc_type);
}

/** \brief Creates the combinations table
  */
void hash_matching::StereoProperties::createCombinations()
{
  vector<string> h_v;
  h_v.push_back("0");
  h_v.push_back("1");

  int d = params_.num_hyperplanes;
  vector< vector<string> > comb_table;
  for (int i=0; i<d; i++)
    comb_table.push_back(h_v);

  vector<string> combinations;
  vector< vector<string> > result;
  recursiveCombinations(comb_table, 0, combinations, result);

  comb_.clear();
  for (uint i=0; i<result.size(); i++)
  {
    string t = "";
    for (uint j=0; j<result[i].size(); j++)
    {
      t += result[i][j];
    }
    comb_.push_back(t);
  }
}

/** \brief Recursive function to create the table of possible combinations
  * given an input vector of vectors.
  * \param vectors containing the possibles values for each parameter
  * \param index of the current iteration
  * \param the result
  */
void hash_matching::StereoProperties::recursiveCombinations(const vector< vector<string> > &all_vecs, 
                                                      size_t vec_idx, vector<string> combinations,
                                                      vector< vector<string> > &result)
{
  if (vec_idx >= all_vecs.size())
  {
    result.push_back(combinations);
    return;
  }
  
  for (size_t i=0; i<all_vecs[vec_idx].size(); i++)
  {
    vector<string> tmp = combinations;
    tmp.push_back(all_vecs[vec_idx][i]);
    recursiveCombinations(all_vecs, vec_idx+1, tmp, result);
  }
}

/** \brief Computes the hyperplanes
 */
void hash_matching::StereoProperties::computeHyperplanes()
{
  // Set the number of hyperplanes
  int d = params_.num_hyperplanes;

  // Compute the descriptors centroid
  centroid_.clear();
  for (int n=0; n<desc_.cols; n++)
  {
    float sum = 0.0;
    for (int m=0; m<desc_.rows; m++)
    {
      float val = desc_.at<float>(m, n);
      sum += val;
    }
    centroid_.push_back(sum / desc_.rows);
  }

  // Generate 'd' random hyperplanes
  srand(101);
  H_.clear();
  for (int i=0; i<d; i++)
  {
    vector<float> h;
    for(int n=0; n<desc_.cols; n++)
    {
      float val = ((float(rand()) / float(RAND_MAX)) * (1 + 1)) -1.0;
      h.push_back(val);
    }
    H_.push_back(h);
  }

  // Make hyperplanes pass through centroid
  delta_.clear();
  for (int i=0; i<d; i++)
  {
    float f = 0.0;
    for(int n=0; n<desc_.cols; n++)
    {
      f -= (float)H_[i][n] * (float)centroid_[n];
    }
    delta_.push_back(f);
  }
}

/** \brief Computes the feature-based hash
 */
void hash_matching::StereoProperties::computeHash()
{
  hash1_.clear();
  hash2_.clear();
  hash3_.clear();

  // Set the number of hyperplanes
  int d = params_.num_hyperplanes;
  
  // Label the feature vectors on either side of the hyperplane as 0 or 1 depending
  // on whether they lie on the left or right side of the hyperplane and create the hash.
  vector< vector<int> > hash_idx;
  for (uint i=0; i<comb_.size(); i++)
  {
    vector<int> t;
    hash_idx.push_back(t);
  }

  vector<uint> hash(comb_.size(), 0);
  for(int m=0; m<desc_.rows; m++)
  {
    string bin = "";
    float desc_mean = 0.0;
    for (int i=0; i<d; i++)
    {
      float v = 0.0;
      for(int n=0; n<desc_.cols; n++)
      {
        v += (float)H_[i][n] * desc_.at<float>(m, n);
        
        // Compute descriptor mean
        if(i == d-1)
          desc_mean += desc_.at<float>(m, n);
      }
      v += delta_[i];

      if(v>0)
        bin += "1";
      else
        bin += "0";
    }

    // Get the position of this bin and increase hash
    int pos = find(comb_.begin(), comb_.end(), bin) - comb_.begin();
    hash[pos]++;

    // Update hash indices
    vector<int> t;
    if (hash_idx[pos].size() != 0)
      t = hash_idx[pos];
    t.push_back(m);
    hash_idx[pos] = t;
  }

  // Add the bins
  hash1_ = hash;

  // Compute the phase between centroid and bin centroids
  for (uint i=0; i<hash_idx.size(); i++)
  {
    // Get the indices for this bin
    vector<int> indices = hash_idx[i];

    // Compute the centroid for every bin
    vector<float> bin_centroid;
    for (int n=0; n<desc_.cols; n++)
    {
      float mean = 0.0;
      for (uint m=0; m<indices.size(); m++)
      {
        mean += desc_.at<float>(indices[m], n);
      }
      mean /= indices.size();
      bin_centroid.push_back(mean);
    }

    // Compute the numerator of the phase
    double num = 0.0;
    double mod = 0.0;
    for (uint j=0; j<bin_centroid.size(); j++)
    {
      num += (double)bin_centroid[j]*(double)centroid_[j];
      mod += pow((double)bin_centroid[j]-(double)centroid_[j], 2);
    }
    mod = sqrt(mod);
    hash3_.push_back(mod);

    // Compute the denominator of the phase
    double mod_a = 0.0;
    double mod_b = 0.0;
    for (uint j=0; j<bin_centroid.size(); j++)
    {
      mod_a += pow((double)bin_centroid[j], 2);
      mod_b += pow((double)centroid_[j], 2);
    }
    double den = sqrt(mod_a) * sqrt(mod_b);

    // Compute the phase
    double phase = acos(num/den);
    hash2_.push_back(phase);
  }
}