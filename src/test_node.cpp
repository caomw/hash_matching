#include <ros/ros.h>
#include "test_base.h"

int main(int argc, char **argv)
{
  ros::init(argc,argv,"test");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  hash_matching::TestBase test(nh,nh_private);

  // Subscription is handled at start and stop service callbacks.
  ros::spin();
  
  return 0;
}