from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='occupancy_grid',
            namespace='occupancy_grid',
            executable='occupancy_grid_node',
            name='sim'
        ),
        Node(
            package='occupancy_grid',
            namespace='simpleFrontier',
            executable='simple_frontier_node',
            name='sim'
        ),

    ])