import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('autodriver_laser_object_segmentation')
    
    # Paths to default files
    default_params_file = os.path.join(pkg_share, 'config', 'params.yaml')
    
    # Launch arguments
    use_cpp_arg = DeclareLaunchArgument(
        'use_cpp_node',
        default_value='true',
        description='Launch C++ node if true; launch Python node if false'
    )
    
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to the parameter configuration YAML file'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (bag) clock if true'
    )

    # Launch Configurations
    use_cpp_node = LaunchConfiguration('use_cpp_node')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # C++ Obstacle Detector Executable Node
    cpp_node = Node(
        package='autodriver_laser_object_segmentation',
        executable='laser_obstacle_detector_node_exe',
        name='laser_obstacle_detector',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        condition=IfCondition(use_cpp_node)
    )

    # Python Obstacle Detector Script Node
    python_node = Node(
        package='autodriver_laser_object_segmentation',
        executable='laser_obstacle_detector.py',
        name='laser_obstacle_detector',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        condition=UnlessCondition(use_cpp_node)
    )

    return LaunchDescription([
        use_cpp_arg,
        params_file_arg,
        use_sim_time_arg,
        cpp_node,
        python_node
    ])
