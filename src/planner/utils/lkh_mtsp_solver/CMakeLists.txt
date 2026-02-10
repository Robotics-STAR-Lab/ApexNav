cmake_minimum_required(VERSION 3.8)
project(lkh_mtsp_solver)

# C++ 표준 설정
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_C_FLAGS "-fcommon")

# 의존성 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 서비스 파일 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SolveMTSP.srv"
  DEPENDENCIES std_msgs
)

# Include 디렉토리 설정
include_directories(
  include
  ${PROJECT_SOURCE_DIR}/include
)

# C 소스 파일 수집
file(GLOB SRCS src/*.c)

# 라이브러리 생성
add_library(lkh_mtsp_solver_lib
  ${SRCS}
  src2/lkh3_interface.cpp
)
target_link_libraries(lkh_mtsp_solver_lib m)

# 실행 파일 생성
add_executable(tsp_node
  ${SRCS}
  src2/tsp_node.cpp
)
target_include_directories(tsp_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(tsp_node rclcpp std_msgs)
target_link_libraries(tsp_node lkh_mtsp_solver_lib m)

# 생성된 서비스 인터페이스 연결
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(tsp_node "${cpp_typesupport_target}")

# 설치
install(TARGETS tsp_node lkh_mtsp_solver_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}/resource
)

# Export
ament_export_include_directories(include)
ament_export_libraries(lkh_mtsp_solver_lib)
ament_export_dependencies(rclcpp std_msgs)

ament_package()