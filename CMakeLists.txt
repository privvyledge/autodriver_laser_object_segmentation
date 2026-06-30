cmake_minimum_required(VERSION 3.8)
project(autodriver_laser_object_segmentation)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(derived_object_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# Include C++ directories
include_directories(include)

# Create C++ Shared Library for Core Logic
add_library(laser_obstacle_detector_core_lib SHARED
  src/laser_obstacle_detector_core.cpp
)
target_include_directories(laser_obstacle_detector_core_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(laser_obstacle_detector_core_lib
  rclcpp
  geometry_msgs
  shape_msgs
  derived_object_msgs
)

# Create ROS 2 Component Node
add_library(laser_obstacle_detector_node SHARED
  src/laser_obstacle_detector_node.cpp
)
target_link_libraries(laser_obstacle_detector_node
  laser_obstacle_detector_core_lib
)
ament_target_dependencies(laser_obstacle_detector_node
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  std_msgs
  shape_msgs
  visualization_msgs
  derived_object_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

# Register component
rclcpp_components_register_node(laser_obstacle_detector_node
  PLUGIN "autodriver_laser_object_segmentation::LaserObstacleDetectorNode"
  EXECUTABLE laser_obstacle_detector_node_exe
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install python scripts
install(PROGRAMS
  scripts/laser_obstacle_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install C++ targets
install(TARGETS
  laser_obstacle_detector_core_lib
  laser_obstacle_detector_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

# Install config and launch directories
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
