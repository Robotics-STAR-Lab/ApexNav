cmake_minimum_required(VERSION 3.8)
project(vis_utils)

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
find_package(visualization_msgs REQUIRED)
find_package(path_searching REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

# Include 디렉토리 설정
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

# 라이브러리 생성
add_library(vis_utils SHARED
  src/planning_visualization.cpp
)

target_include_directories(vis_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(vis_utils
  rclcpp
  std_msgs
  visualization_msgs
  path_searching
  cv_bridge
)

# 설치
install(TARGETS vis_utils
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

# Export
ament_export_include_directories(include)
ament_export_libraries(vis_utils)
ament_export_dependencies(
  rclcpp
  std_msgs
  visualization_msgs
  path_searching
  cv_bridge
)

ament_package()
