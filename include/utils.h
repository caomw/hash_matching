#ifndef UTILS
#define UTILS

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <std_msgs/Float32.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <image_geometry/stereo_camera_model.h>
#include <hash_matching_base.h>

using namespace std;
using namespace cv;

namespace hash_matching
{

class Utils
{

public:

  /** \brief extract the keypoints of some image
    * @return 
    * \param image the source image
    * \param key_points is the pointer for the resulting image key_points
    * \param type descriptor type (see opencv docs)
    */
  static void keypointDetector( const Mat& image, 
                                vector<KeyPoint>& key_points, 
                                string type)
  {
    // Check Opponent color space descriptors
    size_t pos = 0;
    if ( (pos=type.find("Opponent")) == 0)
    {
      pos += string("Opponent").size();
      type = type.substr(pos);
    }

    initModule_nonfree();
    Ptr<FeatureDetector> cv_detector;
    cv_detector = FeatureDetector::create(type);
    try
    {
      cv_detector->detect(image, key_points);
    }
    catch (Exception& e)
    {
      ROS_WARN("[StereoSlam:] cv_detector exception: %s", e.what());
    }
  }

  /** \brief extract descriptors of some image
    * @return 
    * \param image the source image
    * \param key_points keypoints of the source image
    * \param descriptors is the pointer for the resulting image descriptors
    */
  static void descriptorExtraction(const Mat& image,
   vector<KeyPoint>& key_points, Mat& descriptors, string type)
  {
    Ptr<DescriptorExtractor> cv_extractor;
    cv_extractor = DescriptorExtractor::create(type);
    try
    {
      cv_extractor->compute(image, key_points, descriptors);
    }
    catch (Exception& e)
    {
      ROS_WARN("[StereoSlam:] cv_extractor exception: %s", e.what());
    }
  }

  /** \brief match descriptors of 2 images by threshold
    * @return 
    * \param descriptors1 descriptors of image1
    * \param descriptors2 descriptors of image2
    * \param threshold to determine correct matchings
    * \param match_mask mask for matchings
    * \param matches output vector with the matches
    */
  static void thresholdMatching(const Mat& descriptors1, const Mat& descriptors2,
    double threshold, const Mat& match_mask, vector<DMatch>& matches)
  {
    matches.clear();
    if (descriptors1.empty() || descriptors2.empty())
      return;
    assert(descriptors1.type() == descriptors2.type());
    assert(descriptors1.cols == descriptors2.cols);

    const int knn = 2;
    Ptr<DescriptorMatcher> descriptor_matcher;
    // choose matcher based on feature type
    if (descriptors1.type() == CV_8U)
    {
      descriptor_matcher = DescriptorMatcher::create("BruteForce-Hamming");
    }
    else
    {
      descriptor_matcher = DescriptorMatcher::create("BruteForce");
    }
    vector<vector<DMatch> > knn_matches;
    descriptor_matcher->knnMatch(descriptors1, descriptors2,
            knn_matches, knn);

    for (size_t m = 0; m < knn_matches.size(); m++ )
    {
      if (knn_matches[m].size() < 2) continue;
      bool match_allowed = match_mask.empty() ? true : match_mask.at<unsigned char>(
          knn_matches[m][0].queryIdx, knn_matches[m][0].trainIdx) > 0;
      float dist1 = knn_matches[m][0].distance;
      float dist2 = knn_matches[m][1].distance;
      if (dist1 / dist2 < threshold && match_allowed)
      {
        matches.push_back(knn_matches[m][0]);
      }
    }
  }

  /** \brief filter matches of cross check matching
    * @return 
    * \param matches1to2 matches from image 1 to 2
    * \param matches2to1 matches from image 2 to 1
    * \param matches output vector with filtered matches
    */
  static void crossCheckFilter(
      const vector<DMatch>& matches1to2, 
      const vector<DMatch>& matches2to1,
      vector<DMatch>& checked_matches)
  {
    checked_matches.clear();
    for (size_t i = 0; i < matches1to2.size(); ++i)
    {
      bool match_found = false;
      const DMatch& forward_match = matches1to2[i];
      for (size_t j = 0; j < matches2to1.size() && match_found == false; ++j)
      {
        const DMatch& backward_match = matches2to1[j];
        if (forward_match.trainIdx == backward_match.queryIdx &&
            forward_match.queryIdx == backward_match.trainIdx)
        {
          checked_matches.push_back(forward_match);
          match_found = true;
        }
      }
    }
  }

  /** \brief match descriptors of 2 images by threshold
    * @return 
    * \param descriptors1 descriptors of image 1
    * \param descriptors2 descriptors of image 2
    * \param threshold to determine correct matchings
    * \param match_mask mask for matchings
    * \param matches output vector with the matches
    */
  static void crossCheckThresholdMatching(
    const Mat& descriptors1, const Mat& descriptors2,
    double threshold, const Mat& match_mask,
    vector<DMatch>& matches)
  {
    vector<DMatch> query_to_train_matches;
    thresholdMatching(descriptors1, descriptors2, threshold, match_mask, query_to_train_matches);
    vector<DMatch> train_to_query_matches;
    Mat match_mask_t;
    if (!match_mask.empty()) match_mask_t = match_mask.t();
    thresholdMatching(descriptors2, descriptors1, threshold, match_mask_t, train_to_query_matches);

    crossCheckFilter(query_to_train_matches, train_to_query_matches, matches);
  }

  /** \brief Sort 2 trios by hash hash_matching
    * @return true if hash_matching field of trio 1 is smaller than trio 2
    * \param trio 1
    * \param trio 2
    */
  static bool sortTrioByDistance(const hash_matching::HashMatchingBase::trio d1, const hash_matching::HashMatchingBase::trio d2)
  {
    return (d1.hash_matching < d2.hash_matching);
  }

  /** \brief Convert the cvMat type to string
    * @return the string of cvMat type
    * \param type of the input cvMat
    */
  static string matType2str(int type)
  {
    string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    switch ( depth ) {
      case CV_8U:  r = "8U"; break;
      case CV_8S:  r = "8S"; break;
      case CV_16U: r = "16U"; break;
      case CV_16S: r = "16S"; break;
      case CV_32S: r = "32S"; break;
      case CV_32F: r = "32F"; break;
      case CV_64F: r = "64F"; break;
      default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
  }
};

} // namespace

#endif // UTILS


