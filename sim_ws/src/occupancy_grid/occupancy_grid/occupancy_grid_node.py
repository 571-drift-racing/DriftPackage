import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from racecar_interfaces.msg import OccupancyGrid

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class OccupancyGridNode(Node):
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
        self.grid_resolution = 0.1
        self.grid_size = (200, 200)
        self.grid_center = (self.grid_size[0] // 2, self.grid_size[1] // 2)

        # 0: unknown, 1: occupied, -1: free
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.int8)
        
        # Frontier tracking
        self.frontiers = []

        self.publisher_ = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)


        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(
            self.occupancy_grid,
            cmap='gray',
            origin='lower'
        )
        self.ax.set_title("Occupancy Grid")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        plt.ion()  # Enable interactive mode
        plt.show()

        self.get_logger().info("Occupancy Grid Node Initialized")

    def timer_callback(self):
        # Create and populate the custom OccupancyGrid message
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.resolution = self.grid_resolution
        msg.width = self.grid_size[0]
        msg.height = self.grid_size[1]

        # Flatten the 2D grid into a 1D array
        msg.data = self.occupancy_grid.flatten().tolist()

        # Publish the message
        self.publisher_.publish(msg)

    def lidar_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        # self.get_logger().info(f"LIDAR Ranges: {ranges[:10]}")

        angles = np.pi/2 + msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        max_range = msg.range_max

        # Filter invalid points
        valid = (ranges > 0)
        x_coords = ranges[valid] * np.cos(angles[valid])
        y_coords = ranges[valid] * np.sin(angles[valid])

        # Update the grid
        self.update_occupancy_grid(x_coords, y_coords)
        self.update_visualization()

        # Detect frontiers
        # self.frontiers = self.detect_frontiers()


        # self.update_visualization()

    def update_occupancy_grid(self, x_coords, y_coords):
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.int8)
        x_cells = np.floor(x_coords / self.grid_resolution + self.grid_center[0]).astype(int)
        y_cells = np.floor(y_coords / self.grid_resolution + self.grid_center[1]).astype(int)

        x_cells = np.clip(x_cells, 0, self.grid_size[0] - 1)
        y_cells = np.clip(y_cells, 0, self.grid_size[1] - 1)

        # Mark free space
        for x, y in zip(x_cells, y_cells):
            for lx, ly in self.bresenham_line(self.grid_center[0], self.grid_center[1], x, y):
                if self.occupancy_grid[lx, ly] == 0:  # Only update unknown cells to free
                    self.occupancy_grid[lx, ly] = -1  # Free space

        # Mark occupied cells
        self.occupancy_grid[x_cells, y_cells] = 1

        # Debug: Check a subset of the grid
        # self.get_logger().info(f"Grid Center: \n{self.occupancy_grid[self.grid_center[0]-9:self.grid_center[0]+10, self.grid_center[1]-9:self.grid_center[1]+10]}")


    def bresenham_line(self, x0, y0, x1, y1):
        # Bresenham's line algorithm for grid traversal
        line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            line.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return line


    def update_visualization(self):
        # Update the image data in the plot
        self.img.set_data(self.occupancy_grid)
        plt.draw()
        plt.pause(0.001)  # Small pause for rendering


def main(args=None):
    rclpy.init(args=args)
    occupancy_grid_node = OccupancyGridNode()

    try:
        # plt.show(block=False)  # Non-blocking Matplotlib visualization
        rclpy.spin(occupancy_grid_node)
    except KeyboardInterrupt:
        occupancy_grid_node.get_logger().info("Shutting down node.")
    finally:
        # plt.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
