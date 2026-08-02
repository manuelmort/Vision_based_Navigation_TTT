from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument('x', default_value='-9'),
        DeclareLaunchArgument('y', default_value='-4.5'),
        DeclareLaunchArgument('z', default_value='0'),
        DeclareLaunchArgument('yaw', default_value='0'),
    ]

    pkg_share = FindPackageShare('vision_based_navigation_ttt')
    xacro_path = PathJoinSubstitution([pkg_share, 'urdf', 'jackal_gazebo.urdf.xacro'])
    #yaml_file = PathJoinSubstitution([pkg_share, 'config', 'control.yaml'])
    world_path = PathJoinSubstitution([pkg_share, 'GazeboWorlds','obstacle_course.sdf'])

    robot_description = ParameterValue(
        Command(['xacro ', xacro_path]),
        value_type=str
    )

    set_gz_plugin_env = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/opt/ros/jazzy/lib'
    )

    gz_world = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_path],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}, {'use_sim_time': True}],
        output='screen'
    )

    spawn_entity = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_sim', 'create',
            '--name', 'jackal',
            '--x', LaunchConfiguration('x'),
            '--y', LaunchConfiguration('y'),
            '--z', LaunchConfiguration('z'),
            '--Y', LaunchConfiguration('yaw'),
            '--topic', 'robot_description'
        ],
        output='screen'
    )

    spawner_jsb = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_state_broadcaster'],
        output='screen'
    )

    spawner_diffdrive = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'jackal_velocity_controller'],
        output='screen'
    )

    load_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[spawner_jsb],
        )
    )

    load_diffdrive = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawner_jsb,
            on_exit=[spawner_diffdrive],
        )
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",

        ],        
        output='screen'
    )

    optical_flow_node = Node(
        package='vision_based_navigation_ttt',
        executable='optical_flow.py',
        name='optical_flow',
        arguments=['1'],
        parameters=[{'image_sub_name': '/camera/image'}, {'use_sim_time': True}],
        output='screen'
    )
    
    tau_node = Node(
        package='vision_based_navigation_ttt',
        executable='tau_computation.py',
        name='tau_computation',
        parameters=[{'image_sub_name': '/camera/image'}, {'use_sim_time': True}],
        output='screen'
    )

    controller_node = Node(
        package='vision_based_navigation_ttt',
        executable='controller.py',
        name='controller',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    return LaunchDescription(declared_args + [
        set_gz_plugin_env,
        gz_world,
        robot_state_publisher,
        spawn_entity,
        load_jsb,
        bridge,
        load_diffdrive,
        optical_flow_node,
        tau_node,
        controller_node,
    ])
