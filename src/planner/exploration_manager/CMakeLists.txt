cmake_minimum_required(VERSION 3.8)
project(exploration_manager)

# C++ 표준 설정
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

# 의존성 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(plan_env REQUIRED)
find_package(path_searching REQUIRED)
find_package(vis_utils REQUIRED)
find_package(lkh_mtsp_solver REQUIRED)
find_package(trajectory_manager REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(ompl REQUIRED)

# Include 디렉토리 설정
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OMPL_INCLUDE_DIRS}
)

# 실행 파일 생성
add_executable(exploration_node
  src/exploration_node.cpp
  src/exploration_fsm.cpp
  src/exploration_fsm_traj.cpp
  src/exploration_manager.cpp
)

target_include_directories(exploration_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(exploration_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  plan_env
  path_searching
  vis_utils
  lkh_mtsp_solver
  trajectory_manager
  cv_bridge
  tf2
  tf2_ros
)

target_link_libraries(exploration_node
  ${PCL_LIBRARIES}
)

# 설치
install(TARGETS exploration_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Export
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  plan_env
  path_searching
  vis_utils
  lkh_mtsp_solver
  trajectory_manager
  cv_bridge
  tf2
  tf2_ros
)

ament_package()
