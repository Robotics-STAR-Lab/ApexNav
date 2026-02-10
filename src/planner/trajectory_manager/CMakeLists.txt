cmake_minimum_required(VERSION 3.8)
project(trajectory_manager)

# C++ 표준 설정
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

# 의존성 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(plan_env REQUIRED)
find_package(path_searching REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(ompl REQUIRED)
find_package(OsqpEigen REQUIRED)

# 메시지 파일 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PolyTraj.msg"
  DEPENDENCIES std_msgs builtin_interfaces
)

# Include 디렉토리 설정
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OMPL_INCLUDE_DIRS}
)

# traj_server 실행 파일 생성
add_executable(traj_server
  src/traj_server.cpp
  src/mpc.cpp
)

target_include_directories(traj_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(traj_server
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)

target_link_libraries(traj_server
  OsqpEigen::OsqpEigen
)

# 생성된 메시지 인터페이스 연결
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(traj_server "${cpp_typesupport_target}")

# 라이브러리 생성 (메시지 타겟과 이름 충돌 방지를 위해 _lib 접미사 사용)
add_library(trajectory_manager_lib SHARED
  src/optimizer.cpp
)

target_include_directories(trajectory_manager_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(trajectory_manager_lib
  rclcpp
  std_msgs
  geometry_msgs
  visualization_msgs
  plan_env
  path_searching
)

target_link_libraries(trajectory_manager_lib
  ${OMPL_LIBRARIES}
  ${PCL_LIBRARIES}
)

# 설치 - 실행 파일 (export 하지 않음)
install(TARGETS traj_server
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# 설치 - 라이브러리 (export 포함)
install(TARGETS trajectory_manager_lib
  EXPORT trajectory_manager_libExport
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Export
ament_export_include_directories(include)
ament_export_libraries(trajectory_manager_lib)
ament_export_targets(trajectory_manager_libExport HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  std_msgs
  geometry_msgs
  visualization_msgs
  plan_env
  path_searching
)

ament_package()
