import rclpy
from rclpy.node import Node
from racecar_interfaces.msg import OccupancyGrid

import numpy as np

class SimpleFrontierNode(Node):
    def __init__(self):
        super().__init__('simple_frontier_node')

        self.occupancyGrid_sub = self.create_subscription(
            OccupancyGrid,
            '/occupancy_grid',  # Replace with your LIDAR topic name
            self.occupancyGrid_callback,
            100
        )
    

    def occupancyGrid_callback(self, msg: OccupancyGrid):
        # Convert the 1D array back to a 2D grid
        occupancy_grid = np.array(msg.data).reshape((msg.width, msg.height))
        grid_center = (msg.width // 2, msg.height // 2)

        self.get_logger().info(f"Grid Center: \n{occupancy_grid[grid_center[0]-9:grid_center[0]+10, grid_center[1]-9:grid_center[1]+10]}")



def main(args=None):
    rclpy.init(args=args)
    frontierNode = SimpleFrontierNode()

    try:
        # plt.show(block=False)  # Non-blocking Matplotlib visualization
        rclpy.spin(frontierNode)
    except KeyboardInterrupt:
        frontierNode.get_logger().info("Shutting down node.")
    finally:
        # plt.close()
        rclpy.shutdown()