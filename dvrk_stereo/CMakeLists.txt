#
# (C) Copyright 2022 Johns Hopkins University (JHU), All Rights Reserved.
#
# --- begin cisst license - do not edit ---
#
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
#
# --- end cisst license ---

cmake_minimum_required (VERSION 3.1)
project (dvrk_stereo VERSION 2.1.0)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

## find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package (
  catkin REQUIRED
  COMPONENTS
    cv_bridge
    image_transport
    roscpp
    roslib
    sensor_msgs
)

catkin_package()

find_package (OpenCV REQUIRED)

include_directories (include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable (${PROJECT_NAME} src/stereo_proc.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

