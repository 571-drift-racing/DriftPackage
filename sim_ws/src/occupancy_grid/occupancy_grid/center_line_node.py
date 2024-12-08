import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib

class CenterLineNode(Node):
    def __init__(self):
        super().__init__('occupancy_grid_node')

        # LIDAR subscription
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        # Occupancy grid parameters
        self.grid_resolution = 0.05
        self.grid_size = (200, 200)
        self.grid_center = (25, 100) # Y, X

        # 0: unknown, 1: occupied, -1: free
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.int8)
        
        # Visualization setup
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(
            self.occupancy_grid,
            cmap="gray",
            origin="lower",
            extent=[
                0, self.grid_size[0],  # x-axis extent
                0, self.grid_size[1],  # y-axis extent
            ],
        )
        plt.ion()
        plt.show()
        self.get_logger().info("Occupancy Grid Node Initialized")


    def lidar_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)

        angles = -1 * (msg.angle_min + np.arange(len(ranges)) * msg.angle_increment)
        max_range = msg.range_max

        # Filter invalid points
        valid = (ranges > 0) & (ranges < max_range)
        x_coords = ranges[valid] * np.cos(angles[valid])
        y_coords = ranges[valid] * np.sin(angles[valid])

        # Update the grid
        x_coords, y_coords = self.update_occupancy_grid(x_coords, y_coords)
        left_boundary, right_boundary = self.classify_boundaries(x_coords, y_coords)

        left_boundary = left_boundary[np.argsort(np.linalg.norm(left_boundary, axis=1))]
        right_boundary = right_boundary[np.argsort(np.linalg.norm(right_boundary, axis=1))]

        # Long Boundary Should Have More Data Points
        self.b_long = left_boundary if len(left_boundary) > len(right_boundary) else right_boundary
        self.b_short = right_boundary if len(left_boundary) > len(right_boundary) else left_boundary

        self.update_visualization()

    def update_occupancy_grid(self, x_coords, y_coords):
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.int8)
        x_cells = np.floor(x_coords / self.grid_resolution + self.grid_center[0]).astype(int)
        y_cells = np.floor(y_coords / self.grid_resolution + self.grid_center[1]).astype(int)

        x_cells = np.clip(x_cells, 0, self.grid_size[0] - 1)
        y_cells = np.clip(y_cells, 0, self.grid_size[1] - 1)

        return x_cells, y_cells



    def classify_boundaries(self, x_coords, y_coords):
        left, right = [], []
        for x, y in zip(x_coords, y_coords):
            if x < 100:
                left.append((x, y))
            else:
                right.append((x, y))
        return np.array(left), np.array(right)

    def update_visualization(self):
        self.im.set_data(self.occupancy_grid)

        # Plot the boundaries
        self.ax.plot(self.b_long[:, 0], self.b_long[:, 1], 'r-')
        self.ax.plot(self.b_short[:, 0], self.b_short[:, 1], 'b-')

        self.im.autoscale()
        plt.draw()
        plt.pause(0.01)


def main(args=None):
    rclpy.init(args=args)
    center_line_node = CenterLineNode()

    try:
        rclpy.spin(center_line_node)
    except KeyboardInterrupt:
        center_line_node.get_logger().info("Shutting down node.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
