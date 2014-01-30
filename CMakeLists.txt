cmake_minimum_required(VERSION 2.8.3)
project(hash_matching)
find_package(catkin REQUIRED COMPONENTS 
	roscpp
	tf
	cv_bridge
	image_geometry)

# Dependencies - Boost:
find_package(Boost REQUIRED COMPONENTS signals filesystem system)
include_directories(${Boost_INCLUDE_DIRS})

# Dependencies - OpenCV:
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

catkin_package()

include_directories(${catkin_INCLUDE_DIRS} include)

add_executable(hash_matching 
			   src/hash_matching_node.cpp
			   src/hash_matching_base.cpp
			   src/stereo_properties.cpp
			   src/hash.cpp)
target_link_libraries(hash_matching ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
