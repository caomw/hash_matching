#include <ros/ros.h>
#include "hash_matching_base.h"

int main(int argc, char **argv)
{
  ros::init(argc,argv,"hash_matching");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  hash_matching::HashMatchingBase hash_matching(nh,nh_private);
  ros::spin();
  return 0;
}