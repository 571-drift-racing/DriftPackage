import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class OccupancyGridNode(Node):
    def __init__(self):
        super().__init__('occupancy_grid_node')

        # LIDAR subscription
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',  # Replace with your LIDAR topic name
            self.lidar_callback,
            10
        )

        # Occupancy grid parameters
        self.grid_resolution = 0.15  # 25cm per cell
        self.grid_size = (200, 200)  # 50m x 50m grid
        self.grid_center = (self.grid_size[0] // 2, self.grid_size[1] // 2)
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.int8)  # 0: unknown, 1: occupied, -1: free
        
        # Frontier tracking
        self.frontiers = []

        # Set up Matplotlib visualization
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.occupancy_grid, cmap='hot', origin='lower', vmin=-1, vmax=1, extent=[-10, 10, -10, 10])
        self.ax.set_title("Occupancy Grid with Frontiers")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.frontier_points, = self.ax.plot([], [], 'ro', markersize=2, label="Frontiers")
        self.ax.legend()

        # Timer for Matplotlib updates
        self.anim = FuncAnimation(self.fig, self.update_visualization, interval=100)

        self.get_logger().info("Occupancy Grid Node Initialized with Matplotlib Visualization")


    def print_points(self, x, y):
        for i in range(len(x)):
            print(f"({x[i]}, {y[i]})", end=" ")
            if i % 8 == 0:
                print()

    def lidar_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        self.get_logger().info(f"LIDAR Ranges: {ranges[:10]}")

        angles = np.pi/2 + msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        max_range = msg.range_max

        # Filter invalid points
        valid = (ranges > 0)
        x_coords = ranges[valid] * np.cos(angles[valid])
        y_coords = ranges[valid] * np.sin(angles[valid])

        # self.ax.clear()
        # self.ax.scatter(x_coords, y_coords, s=1, c='blue')
        # self.ax.set_title('LIDAR Point Cloud')
        # self.ax.set_xlabel('X Coordinates (meters)')
        # self.ax.set_ylabel('Y Coordinates (meters)')
        # self.ax.axis('equal')
        # self.ax.grid(True)

        # plt.draw()
        # plt.pause(0.001)

        # Update the grid
        self.update_occupancy_grid(x_coords, y_coords)

        # Detect frontiers
        self.frontiers = self.detect_frontiers()

    def update_occupancy_grid(self, x_coords, y_coords):
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
        self.get_logger().info(f"Grid Center (5x5): \n{self.occupancy_grid[self.grid_center[0]-9:self.grid_center[0]+10, self.grid_center[1]-9:self.grid_center[1]+10]}")

    def detect_frontiers(self):
        frontiers = []
        for x in range(1, self.grid_size[0] - 1):
            for y in range(1, self.grid_size[1] - 1):
                if self.occupancy_grid[x, y] == -1:  # Free cell
                    neighbors = self.occupancy_grid[x-1:x+2, y-1:y+2].flatten()
                    if 0 in neighbors:  # If there's an unknown cell nearby
                        frontiers.append((x, y))

        # Debug: Check frontier count
        self.get_logger().info(f"Detected {len(frontiers)} frontiers")
        return frontiers

    def update_visualization(self, _):
        """
        Update the Matplotlib visualization dynamically.
        """
        # Update the occupancy grid display
        self.img.set_data(self.occupancy_grid)

        # Convert frontier grid coordinates to physical space
        frontier_x = [(x - self.grid_center[0]) * self.grid_resolution for x, y in self.frontiers]
        frontier_y = [(y - self.grid_center[1]) * self.grid_resolution for x, y in self.frontiers]

        # Update frontier points
        self.frontier_points.set_data(frontier_x, frontier_y)

        return [self.img, self.frontier_points]

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


def main(args=None):
    rclpy.init(args=args)
    occupancy_grid_node = OccupancyGridNode()

    try:
        plt.show(block=False)  # Non-blocking Matplotlib visualization
        rclpy.spin(occupancy_grid_node)
    except KeyboardInterrupt:
        occupancy_grid_node.get_logger().info("Shutting down node.")
    finally:
        plt.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
