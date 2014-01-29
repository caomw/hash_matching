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
  bucket_max(DEFAULT_BUCKET_MAX),
  features_max_value(DEFAULT_F_MAX_VALUE),
  N_levels(DEFAULT_N_LEVELS)
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
vector<uint> hash_matching::StereoProperties::getHash1() { return hash1_; } // feature location with respect hyperplanes
vector<double> hash_matching::StereoProperties::getHash2() { return hash2_; } // phase | bin centroid-planes centroid|
vector<double> hash_matching::StereoProperties::getHash3() { return hash3_; } // module | bin centroid-planes centroid|
vector<double> hash_matching::StereoProperties::getHash4() { return hash4_; } // centroid variance
vector<uint> hash_matching::StereoProperties::getHash5() { return hash5_; } // hystogram feature distribution
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
  hash4_.clear();

  // Set the number of hyperplanes
  int d = params_.num_hyperplanes;
  
  // Label the feature vectors on either side of the hyperplane as 0 or 1 depending
  // on whether they lie on the left or right side of the hyperplane and create the hash.
  vector< vector<int> > hash_idx;
  vector< vector<float> > bin_centroid_container; // container of bin centroids
  vector <double> dispersionvector(desc_.cols,0); 
  
  for (uint i=0; i<comb_.size(); i++)
  {
    vector<int> t;
    hash_idx.push_back(t);
  }

  for (uint i=0; i<comb_.size(); i++) // one centroid for area (if there are no features, empty)
  {
    vector<float> t;
    bin_centroid_container.push_back(t);
  }


  vector<uint> hash(comb_.size(), 0); // 16 possible combinations, one hash for each combination initilaized to 0
  for(int m=0; m<desc_.rows; m++) // for each descriptor
  {
    string bin = "";
    float desc_mean = 0.0;
    for (int i=0; i<d; i++) // for each plane 
    {
      float v = 0.0;
      for(int n=0; n<desc_.cols; n++) // for each component of the descriptor
      {
        if (desc_.at<float>(m, n)>255 || desc_.at<float>(m, n) < 0)
        ROS_INFO_STREAM("feature " << m << "=" << desc_.at<float>(m, n));

        v += (float)H_[i][n] * desc_.at<float>(m, n); // plane "i" column "n" (variable ieesim of the plane) 
        // * descriptor m column "n", we are applying the plane equation to the descriptor
        
        // Compute descriptor mean
        if(i == d-1)
          desc_mean += desc_.at<float>(m, n);
      }
      v += delta_[i];  // the delta value is the constant of the plane equation. 

      if(v>0)
        bin += "1"; // accumulate the set of bits, one per feature/plane
      else
        bin += "0";
    }

    // Get the position of this bin and increase hash
    int pos = find(comb_.begin(), comb_.end(), bin) - comb_.begin(); //  find the position of the 4 bit bin
    // in the combinations matrix  0<pos<16 per 4 plans, 64 per 6 plans
    
    if (find(comb_.begin(), comb_.end(), bin)!=comb_.end()) // if the combination exists, in principle, the combination should always exists.
    {
      hash[pos]++; // increment the hash index if found. hash contains the quantity of occurences of certain 
    // N bits bin , which is the same of how many features present the same N bin combination, 
    //so are located in the same region delimited by the multiple planes. 

    // Update hash indices, this structure stores the index of the hash corresponding to descriptor "m"
      vector<int> t;
      if (hash_idx[pos].size() != 0) // if this bin has already other features assigned
         t = hash_idx[pos]; 
      t.push_back(m); // add to vector t the descriptor with index "m", 
      hash_idx[pos] = t; // the hash index of the bin combination "pos" is a vector of "m" descriptors 
    // that have the same bin, there are 16 hashes corresponding to 
    // 16 different combinations of 4 bits
    }
  }

  // Add the bins
  hash1_ = hash;

  // compute the 5th hash : the feature quantization hystogram
  double quantification_interval=(params_.features_max_value/params_.N_levels);
  ROS_INFO_STREAM("quantification_interval;" << (float)quantification_interval << "valores: " << params_.features_max_value << ";" << params_.N_levels);  
  ROS_INFO_STREAM("resto quatifica;" << fmod(params_.features_max_value,params_.N_levels));  
  int level, integer_part=0; 
  vector<uint> hystogram(params_.N_levels, 0); // vector of integers, it contains the hystogram values (number of occurences) for each level. 
  // initialize the hystogram with 0's

  for(int m=0; m<desc_.rows; m++) // for each descriptor
  {
    for(int n=0; n<desc_.cols; n++) // for each component of the descriptor
      {
        float resto=fmodf(desc_.at<float>(m, n),(float)quantification_interval); // divide the feature value by the q. interval
        integer_part = (int)(desc_.at<float>(m, n)/(float)quantification_interval); // and take the remainder and the integer part.  
        if (resto>0) // if the remainder is dif. from 0, 
        {
          level=integer_part+1; // quantification level
          hystogram[level]++; // account for occurences of feature components with a value in a certain level. 
          //ROS_INFO_STREAM("resto;" << resto << "parte entera " << level <<  "valores: " << desc_.at<float>(m, n) << ";" << quantification_interval);  
        }
        else 
        {
          level=integer_part;
          hystogram[level]++;
        }
      }
  }
  hash5_=hystogram;


  // Compute the phase between centroid and bin centroids
  for (uint i=0; i<hash_idx.size(); i++) // for each bin
  {
    // Get the indices for this bin
    vector<int> indices = hash_idx[i]; // this vector contains all features (set of m's) that have the bin "i"
    vector<float> bin_centroid;
    // Compute the centroid for every bin, the centroid of a set of features that have the same bin.
    //vector<float> bin_centroid;
    double num = 0.0;
    double mod = 0.0;
    double phase=0;
    if (indices.size()!=0) // if this combination has at least one feature associated.....(index not empty)
      //calculate the centroid, its module and phane and store them.
    {
      for (int n=0; n<desc_.cols; n++) // for every column of the descriptors
      {
        float mean = 0.0;
        for (uint m=0; m<indices.size(); m++)  
        {
          mean += desc_.at<float>(indices[m], n); // search for the feature "m" with bin "i" and accumulate
        }
        mean /= indices.size(); // divide by the number of features with bin "i"
       // ROS_INFO_STREAM("pet4:" << mean);
      //ROS_INFO_STREAM("mean" << mean << "n" << n);
        bin_centroid.push_back(mean); // we have calculated the mean, column by column, of all descriptors 
      //ROS_INFO_STREAM("bin centroid size" << bin_centroid.size());
      // with the same bin (bit combination) --> centroid of a bin.
      }
      bin_centroid_container[i]=bin_centroid;

    // Compute the numerator of the phase
      
      for (uint j=0; j<bin_centroid.size(); j++) // bin centroid size = 128
      {
        num += (double)bin_centroid[j]*(double)centroid_[j]; // centroid is unique, 
        mod+=pow((double)bin_centroid[j]-(double)centroid_[j], 2); // diff. between centroid and the iessim bin
      // ROS_INFO_STREAM("mod" << mod << "a" << a << "j" << j);
      }
      mod = sqrt(mod); // store in hash3 the module of the distance between the centroid and the bin_centroid

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
    phase = acos(num/den);
    // ROS_INFO_STREAM("phase" << phase);
    } // if that hash index does not contain any feature, do nothing.
    hash3_.push_back(mod); // if the bin has no indexes, store 0. 
    hash2_.push_back(phase);
  }
        

  double dispersionvectormodule=0.0;

  // compute the deviation (varianze) of a group of descriptors that share the same bin with respect to their centroid
  for (uint i=0; i<hash_idx.size(); i++) // for each bin
  {
    // Get the indices for this bin
    vector<int> indices = hash_idx[i]; // this vector contains the indexes of all features (set of m's) that have the bin "i"
    
    vector <double> accumulateddiff(desc_.cols,0); // vector<uint> hash(comb_.size(), 0);
    if (indices.size()!=0)
    { 
      vector<float> bin_centroid;
      bin_centroid=bin_centroid_container[i]; // retrieve the centroid (vectors of 128 floats) of bin "i"
      for (uint j=0; j<indices.size(); j++) // for each descriptor
      {
        for(int n=0; n<desc_.cols; n++) // for each component of the descriptor 0<n<128
        {
          double a=(double)desc_.at<float>(indices[j],n);
          double b=(double)bin_centroid[n];
        //ROS_INFO_STREAM("a y b" << a << ";" << b << "j" << j);   
          accumulateddiff[n]+=pow((a-b),2); // compute the squared diff. of component "n" and accumulate 
        //ROS_INFO_STREAM("pow((a-b),2)" << pow((a-b),2) << "n" << n << "j" << j);   
        }
      } // the variance (dispersion) is:  summation ( pow((xi-Xc),2) )/n-1 , since we have a vector , we can generate 
    // a vector of dispersions, one at each direction.

      for (uint n=0; n<accumulateddiff.size(); n++) // for each component of the accumulated squared diff. 
      {
        dispersionvector[n] = accumulateddiff[n]/indices.size(); // divide by the number of descriptors of this bin
        dispersionvectormodule+=pow(dispersionvector[n],2);
      // ROS_INFO_STREAM("diff" << n);
      }
      dispersionvectormodule=sqrt(dispersionvectormodule); // module of the dispersion vector 
    //ROS_INFO_STREAM("dispersionvectormodule" << dispersionvectormodule);
       // store a variance for each bin
    }
    hash4_.push_back(dispersionvectormodule);
  }
} // at this point we have 4 hash components for each bin: 1) the number of features that have the same bin, the
// 2) the module of the bin centroid with respect to the global centroid, 3) the phase of the bin centroid with 
// respect to the global centroid, 4) the dispersion of all features sharing the same bin with respect to the 
// bin centroid.