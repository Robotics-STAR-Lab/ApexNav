cmake_minimum_required(VERSION 3.8)
project(path_searching)

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
find_package(plan_env REQUIRED)
find_package(cv_bridge REQUIRED)

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

# 라이브러리 생성
add_library(path_searching SHARED
  src/astar2d.cpp
  src/kino_astar.cpp
)

target_include_directories(path_searching PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(path_searching
  rclcpp
  std_msgs
  visualization_msgs
  plan_env
  cv_bridge
)

target_link_libraries(path_searching
  ${OMPL_LIBRARIES}
  ${PCL_LIBRARIES}
)

# 설치
install(TARGETS path_searching
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

# Export
ament_export_include_directories(include)
ament_export_libraries(path_searching)
ament_export_dependencies(
  rclcpp
  std_msgs
  visualization_msgs
  plan_env
  cv_bridge
)

ament_package()
