cmake_minimum_required(VERSION 3.0.2)
project(state_monitor)

find_package(ftxui 5 REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES state_monitor
  CATKIN_DEPENDS roscpp rospy std_msgs
  #  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(state_monitor_node
  src/state_monitor.cpp
)

target_link_libraries(state_monitor_node
  PRIVATE ftxui::screen
  PRIVATE ftxui::dom
  PRIVATE ftxui::component # Not needed for this example.
  ${catkin_LIBRARIES}
)
