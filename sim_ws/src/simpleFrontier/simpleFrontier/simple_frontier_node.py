import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from racecar_interfaces.msg import OccupancyGrid

import numpy as np

class SimpleFrontierNode(Node):
    def __init__(self):
        super().__init__('simple_frontier_node')

        self.occupancyGrid_sub = self.create_subscription(
            OccupancyGrid,
            '/occupancy_grid',  # Replace with your LIDAR topic name
            self.occupancyGrid_callback,
            1000
        )

        self.cmdDrive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10)

        self.occupancy_grid = None
        self.grid_size = None

        self.get_logger().info("Frontier Node Initialized")
    

    def occupancyGrid_callback(self, msg: OccupancyGrid):
        # Convert the 1D array back to a 2D grid
        self.occupancy_grid = np.array(msg.data).reshape((msg.width, msg.height))
        if not self.grid_size:
            self.grid_size = (msg.width, msg.height)

        grid_center = (msg.width // 2, msg.height // 2)

        # self.get_logger().info(f"Grid Center: \n{self.occupancy_grid[grid_center[0]-9:grid_center[0]+10, grid_center[1]-9:grid_center[1]+10]}")
        self.detect_frontiers()


    def detect_frontiers(self):
        frontiers = []
        avg_x, avg_y = 0 ,0
        for x in range(1, self.grid_size[0] - 1):
            for y in range(1, self.grid_size[1] - 1):
                if self.occupancy_grid[x, y] == -1:  # Free cell
                    neighbors = self.occupancy_grid[x-1:x+2, y-1:y+2].flatten()
                    if 0 in neighbors:  # If there's an unknown cell nearby
                        frontiers.append((x, y))
                        avg_x, avg_y = avg_x + x, avg_y + y

        # driveMsg = AckermannDriveStamped()
        # driveMsg.drive.steering_angle = avg_x/len(frontiers) - 100
        # driveMsg.drive.speed = (avg_y/len(frontiers) - 100 / 100) * 7.0
        # self.cmdDrive_pub.publish(driveMsg)

    

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