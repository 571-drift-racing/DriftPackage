import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from scipy.interpolate import interp1d
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
        self.grid_center = (100, 25)

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

        angles = np.pi/2 + msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        max_range = msg.range_max

        # Filter invalid points
        valid = (ranges > 0) & (ranges < max_range)
        x_coords = ranges[valid] * np.cos(angles[valid])
        y_coords = ranges[valid] * np.sin(angles[valid])

        x_coords, y_coords = self.update_occupancy_grid(x_coords, y_coords)
        self.update_boundary(x_coords, y_coords)
        self.update_visualization()

    def update_occupancy_grid(self, x_coords, y_coords):
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.int8)
        x_cells = np.floor(x_coords / self.grid_resolution + self.grid_center[0]).astype(int)
        y_cells = np.floor(y_coords / self.grid_resolution + self.grid_center[1]).astype(int)

        x_cells = np.clip(x_cells, 0, self.grid_size[0] - 1)
        y_cells = np.clip(y_cells, 0, self.grid_size[1] - 1)

        # Mark free space
        # for x, y in zip(x_cells, y_cells):
        #     for lx, ly in self.bresenham_line(self.grid_center[0], self.grid_center[1], x, y):
        #         if self.occupancy_grid[ly, lx] == 0:  # Only update unknown cells to free
        #             self.occupancy_grid[ly, lx] = -1  # Free space

        # # Mark occupied cells
        # self.occupancy_grid[y_cells, x_cells] = 1
        return x_cells, y_cells
    
    def bresenham_line(self, x0, y0, x1, y1):
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


    def update_boundary(self, x_coords, y_coords):
        left_boundary, right_boundary = self.classify_boundaries(x_coords, y_coords)

        if len(left_boundary) < 20 or len(right_boundary) < 20:
            return

        left_length = np.sum(np.sqrt(np.sum(np.diff(left_boundary, axis=0)**2, axis=1)))
        right_length = np.sum(np.sqrt(np.sum(np.diff(right_boundary, axis=0)**2, axis=1)))

        if left_length > right_length:
            self.b_long, self.b_short = left_boundary, right_boundary
        else:
            self.b_long, self.b_short = right_boundary, left_boundary

        self.b_long = self.resample_boundary(self.b_long, 100)
        self.b_short = self.resample_boundary(self.b_short, 50)

        self.segments = self.calculate_segments(self.b_long, self.b_short, 55)

    def classify_boundaries(self, x_coords, y_coords):
        # Combine x and y coordinates into a single array
        points = np.column_stack((x_coords, y_coords))

        # Initialize boundaries
        left, right = [], []
        is_storing_right = True

        avg_delta = 0
        alpha = 0.90

        # Iterate through the points
        for point in points:
            if is_storing_right:
                if len(right) > 0:
                    delta = np.linalg.norm(point - right[-1])
                    avg_delta = alpha * delta + (1 - alpha) * avg_delta
                    
                if len(right) < 10:
                    right.append(point)
                    
                else:
                    if abs(delta - avg_delta) < 0.8:
                        right.append(point)
                    else:
                        left.append(point)
                        is_storing_right = False

            else:
                left.append(point)

        # Convert to numpy arrays
        return np.array(left), np.array(right)
    
    def resample_boundary(self, boundary, points):
        distances = np.cumsum(np.sqrt(np.sum(np.diff(boundary, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)

        interp_func = interp1d(distances, boundary, kind='linear', axis=0)
        new_distances = np.linspace(0, distances[-1], points)
        return interp_func(new_distances)

    def calculate_segments(self, b_long, b_short, w_max):
        """
        Calculate segments based on b_long and b_short boundaries.
        
        Args:
            b_long (np.ndarray): Resampled long boundary (N x 2).
            b_short (np.ndarray): Resampled short boundary (M x 2).
            w_max (float): Maximum allowed track width for matching.
        
        Returns:
            segments (list): List of matched and projected segments.
        """
        segments = []
        n_long = len(b_long)
        
        for i in range(n_long):
            # Step 1: Compute distances from b_long[i] to all points in b_short
            distances = np.linalg.norm(b_short - b_long[i], axis=1)
            k_i = np.argmin(distances)  # Find the nearest point in b_short
            min_distance = distances[k_i]

            if min_distance < w_max:
                # Step 2: Match point if within track width
                segments.append([b_long[i], b_short[k_i]])
            else:
                # Step 3: Project remaining long boundary
                remaining_b_long = b_long[i:]
                
                if len(segments) > 0:
                    matched_distances = [np.linalg.norm(s[0] - s[1]) for s in segments]
                    avg_width = np.mean(matched_distances)
                else:
                    avg_width = w_max

                # Calculate normals for projection
                normals = self.calculate_normals(remaining_b_long)

                # Append projected segment to the list
                for j, point in enumerate(remaining_b_long):
                    projected_point = point + normals[j] * avg_width
                    if np.linalg.norm(projected_point - point) <= 1.5 * w_max:  # Keep projection within bounds
                        segments.append([point, projected_point])
                    
                break  # End loop after projecting the remaining section

        return segments
    
    def calculate_normals(self, points):
        """
        Calculate approximate normals for a set of points.
        
        Args:
            points (np.ndarray): Array of points (N x 2).
        
        Returns:
            np.ndarray: Array of normal vectors (N x 2).
        """
        if len(points) < 2:
            raise ValueError("At least two points are required to calculate normals.")
        
        normals = []
        n = len(points)

        for i in range(n):
            if i == 0:  # First point
                tangent = points[i + 1] - points[i]
            elif i == n - 1:  # Last point
                tangent = points[i] - points[i - 1]
            else:  # Interior points
                tangent = points[i + 1] - points[i - 1]

            # Normalize the tangent vector
            norm = np.linalg.norm(tangent)
            if norm > 0:
                tangent /= norm
                # Rotate tangent vector 90 degrees to get the normal
                normal = np.array([-tangent[1], tangent[0]])
            else:
                normal = np.array([0, 0])  # Default normal if tangent is degenerate

            normals.append(normal)

        # Convert normals to numpy array
        normals = np.array(normals)

        return normals

    def update_visualization(self):
        # Clear the axes
        self.ax.clear()

        # Re-plot the occupancy grid
        # self.im = self.ax.imshow(
        #     self.occupancy_grid,
        #     cmap="gray",
        #     origin="lower",
        #     extent=[
        #         0, self.grid_size[0],  # x-axis extent
        #         0, self.grid_size[1],  # y-axis extent
        #     ],
        # )

        # Plot the boundaries
        self.ax.plot(self.b_long[:, 0], self.b_long[:, 1], 'r-', label="Long Boundary")
        self.ax.plot(self.b_short[:, 0], self.b_short[:, 1], 'b-', label="Short Boundary")

        if hasattr(self, 'segments'):  # Ensure `segments` exists
            for segment in self.segments:
                # Each segment contains two points: [point_long, point_short]
                point_long, point_short = segment
                self.ax.plot(
                    [point_long[0], point_short[0]],
                    [point_long[1], point_short[1]],
                    'g-'  # Green line for the segment
                )

        # Optional: Add legend
        self.ax.legend()

        # Update the visualization
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
