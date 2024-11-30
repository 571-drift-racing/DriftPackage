import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import numpy as np


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

        # Publishers
        self.grid_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)
        self.frontier_pub = self.create_publisher(Marker, '/frontiers', 10)

        # Occupancy grid parameters
        self.grid_resolution = 0.05  # 5cm per cell
        self.grid_size = (200, 200)  # 10m x 10m grid
        self.grid_center = (self.grid_size[0] // 2, self.grid_size[1] // 2)
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.int8)  # 0: unknown, 1: occupied, -1: free
        
        # Frontier tracking
        self.frontiers = []

        self.get_logger().info("Occupancy Grid Node Initialized")

    def lidar_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        max_range = msg.range_max
        
        # Convert LIDAR data to Cartesian coordinates
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)

        # Filter invalid data
        valid = (ranges > 0) & (ranges < max_range)
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]

        # Update occupancy grid
        self.update_occupancy_grid(x_coords, y_coords)

        # Detect frontiers
        self.frontiers = self.detect_frontiers()

        # Publish the grid and frontiers
        self.publish_occupancy_grid()
        self.publish_frontiers()

    def update_occupancy_grid(self, x_coords, y_coords):
        # Map coordinates to grid cells
        x_cells = np.floor(x_coords / self.grid_resolution + self.grid_center[0]).astype(int)
        y_cells = np.floor(y_coords / self.grid_resolution + self.grid_center[1]).astype(int)

        # Clamp to grid bounds
        x_cells = np.clip(x_cells, 0, self.grid_size[0] - 1)
        y_cells = np.clip(y_cells, 0, self.grid_size[1] - 1)

        # Mark occupied cells
        self.occupancy_grid[x_cells, y_cells] = 1

        # Mark free space (along rays)
        for x, y in zip(x_cells, y_cells):
            for lx, ly in self.bresenham_line(self.grid_center[0], self.grid_center[1], x, y):
                self.occupancy_grid[lx, ly] = -1

    def detect_frontiers(self):
        frontiers = []
        for x in range(1, self.grid_size[0] - 1):
            for y in range(1, self.grid_size[1] - 1):
                if self.occupancy_grid[x, y] == -1:  # Free cell
                    # Check if neighbors include unknown cells
                    neighbors = self.occupancy_grid[x-1:x+2, y-1:y+2].flatten()
                    if 0 in neighbors:
                        frontiers.append((x, y))
        return frontiers

    def publish_occupancy_grid(self):
        """
        Publish the occupancy grid as a ROS2 `nav_msgs/OccupancyGrid` message.
        """
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'

        # Grid metadata
        grid_msg.info.resolution = self.grid_resolution
        grid_msg.info.width = self.grid_size[0]
        grid_msg.info.height = self.grid_size[1]
        grid_msg.info.origin.position.x = -self.grid_size[0] * self.grid_resolution / 2
        grid_msg.info.origin.position.y = -self.grid_size[1] * self.grid_resolution / 2
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # Flatten grid and convert to ROS-compatible format
        flattened_grid = self.occupancy_grid.flatten()
        grid_msg.data = [int(cell) for cell in flattened_grid]

        # Publish the grid
        self.grid_pub.publish(grid_msg)

    def publish_frontiers(self):
        """
        Publish frontiers as a `visualization_msgs/Marker` message.
        """
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'

        marker.ns = 'frontiers'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.scale.x = 0.1  # Point size
        marker.scale.y = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque

        # Add frontier points
        for (x, y) in self.frontiers:
            # Convert grid coordinates to physical space
            point = Point()
            point.x = (x - self.grid_center[0]) * self.grid_resolution
            point.y = (y - self.grid_center[1]) * self.grid_resolution
            point.z = 0.0
            marker.points.append(point)

        # Publish marker
        self.frontier_pub.publish(marker)

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
        rclpy.spin(occupancy_grid_node)
    except KeyboardInterrupt:
        occupancy_grid_node.get_logger().info("Shutting down node.")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
