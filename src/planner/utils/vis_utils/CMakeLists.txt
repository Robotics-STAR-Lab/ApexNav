cmake_minimum_required(VERSION 2.8.3)
project(vis_utils)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
    path_searching
    roscpp
    std_msgs
    cv_bridge
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES vis_utils
    CATKIN_DEPENDS path_searching
    #  DEPENDS system_lib
)

include_directories(
    SYSTEM
    include
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

add_library(vis_utils
    src/planning_visualization.cpp
)
target_link_libraries(vis_utils
    ${catkin_LIBRARIES}
)