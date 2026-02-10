cmake_minimum_required(VERSION 3.8)
project(plan_env)

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
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

# 메시지 파일 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MultipleMasksWithConfidence.msg"
  DEPENDENCIES std_msgs sensor_msgs
)

# Include 디렉토리 설정
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

# 라이브러리 생성
add_library(plan_env_lib SHARED
  src/sdf_map2d.cpp
  src/map_ros.cpp
  src/raycast2d.cpp
  src/perception_utils2d.cpp
  src/frontier_map2d.cpp
  src/object_map2d.cpp
  src/value_map2d.cpp
)

target_include_directories(plan_env_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(plan_env_lib
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  cv_bridge
  message_filters
  pcl_conversions
  tf2
  tf2_ros
)

target_link_libraries(plan_env_lib
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# 생성된 메시지 인터페이스 연결
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(plan_env_lib "${cpp_typesupport_target}")

# 설치
install(TARGETS plan_env_lib
  EXPORT plan_env_libExport
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

# Export
ament_export_include_directories(include)
ament_export_libraries(plan_env_lib)
ament_export_targets(plan_env_libExport HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  cv_bridge
  message_filters
  pcl_conversions
  tf2
  tf2_ros
)

ament_package()
