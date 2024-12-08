import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from racecar_interfaces.msg import OccupancyGrid

import numpy as np
import math
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
            self.grid_center = (msg.width // 2, msg.height // 2)

        # self.get_logger().info(f"Grid Center: \n{self.occupancy_grid[0:5, 0:5]}")
        self.explore_frontiers()


    def detect_frontiers(self):
        frontiers = []
        avg_x, avg_y = 0 ,0
        for x in range(110, self.grid_size[0] - 1):
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
        return frontiers
    def select_closest_frontier(self, frontiers):
        if not frontiers:
            return None

        grid_center = (self.grid_size[0] // 2, self.grid_size[1] // 2)

        min_distance = float('inf')
        target_frontier = None

        for frontier in frontiers:
            dx = frontier[0] - grid_center[0]
            dy = frontier[1] - grid_center[1]
            distance = math.sqrt(dx**2 + dy**2)
            if dx> 0 and distance < min_distance:
                min_distance = distance
                target_frontier = frontier
                
        return target_frontier
    
    def calculate_motion_command(self, target):
        dx = target[0] - self.grid_center[0]
        dy = target[1] - self.grid_center[1]
        angle_to_target = -1 * math.atan2(dy, dx)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = angle_to_target
        drive_msg.drive.speed = 3.0

        return drive_msg
    
    def is_path_clear(self, drive_msg):
        grid_center = (self.grid_size[0] // 2, self.grid_size[1] // 2)
        dx = int(drive_msg.drive.steering_angle * 10)  
        dy = int(drive_msg.drive.speed * 10)  

        x, y = grid_center
        while 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            if self.occupancy_grid[x, y] == 1:  
                return False
            x += dx
            y += dy

        return True
    
    def stop_robot(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = 0.0
        drive_msg.drive.speed = 0.0
        self.cmdDrive_pub.publish(drive_msg)
        self.get_logger().info("Robot stopped.")
    
    def explore_frontiers(self):
        frontiers = self.detect_frontiers()

        if not frontiers:
            self.get_logger().info("No frontiers detected. Exploration complete.")
            return

        target_frontier = self.select_closest_frontier(frontiers)

        print(target_frontier)
        drive_msg = self.calculate_motion_command(target_frontier)
        self.cmdDrive_pub.publish(drive_msg)


        # if target_frontier:
        #     drive_msg = self.calculate_motion_command(target_frontier)

        #     self.get_logger().info("Starting PathClear()")
        #     if self.is_path_clear(drive_msg):
        #         self.cmdDrive_pub.publish(drive_msg)
        #         self.get_logger().info(f"Driving to frontier at {target_frontier}")
        #     else:
        #         self.get_logger().warn("Path blocked! Re-evaluating frontiers.")
        #         self.stop_robot()
        # else:
        #     self.get_logger().warn("No valid target frontier found.")

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

if __name__ == '__main__':
    main()
