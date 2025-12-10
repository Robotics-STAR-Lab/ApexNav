cmake_minimum_required(VERSION 3.0.2)
project(trajectory_manager)

set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  plan_env
  roscpp
  sensor_msgs
  std_msgs
  visualization_msgs
  path_searching
  message_generation
)

find_package(OsqpEigen REQUIRED)

add_message_files(
  FILES
  PolyTraj.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(ompl REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES trajectory_manager
  CATKIN_DEPENDS path_searching geometry_msgs plan_env roscpp sensor_msgs std_msgs visualization_msgs message_runtime
  DEPENDS system_lib
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OMPL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  include
)

add_executable(traj_server
  src/traj_server.cpp
  src/mpc.cpp
)
target_link_libraries(traj_server
  ${catkin_LIBRARIES}
  OsqpEigen::OsqpEigen
)

add_library(trajectory_manager
  src/optimizer.cpp
)
