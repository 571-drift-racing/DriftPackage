import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from racecar_interfaces.msg import DriftData, CenterLine

from scipy.interpolate import interp1d, splprep, splev
import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib
import time

############## Constants ##############
wheelbase = .3302 #car length meters
l_c, l_b = 0.75, 2  # Lookahead Parameters

boundaryDeltaThreshold = 0.8 # Threshold for classifying boundaries
resampleBoundary_short, resampleBoundary_long = 50, 100
segmentMaxLength = 50
############## Constants ##############

DEBUG = False

class CenterLineNode(Node):
    def __init__(self):
        super().__init__('occupancy_grid_node')

        self.heading = None
        self.curSpeed = 0
        self.isLeftLong = None

        self.prevTime = None
        self.prevHeading = None


        # LIDAR subscription
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            40
        )

        self.driftData_pub = self.create_publisher(
            DriftData,
            '/drift_data',
            10
        )
        self.centerLine_pub = self.create_publisher(
            CenterLine,
            '/center_line',
            10
        )

        # Occupancy grid parameters
        self.grid_resolution = 0.05
        self.grid_size = (200, 200)
        self.grid_center = (100, 25)

        # 0: unknown, 1: occupied, -1: free
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.int8)
        
        if DEBUG:
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


    def publishDrift(self):
        if self.centerline is None:
            return

        middlePoint = self.centerline[len(self.centerline)//2]
        frontierPoint = self.centerline[-3]

        msg = DriftData()
        msg.middle_point.x = middlePoint[0]
        msg.middle_point.y = middlePoint[1]
        msg.frontier_point.x = frontierPoint[0]
        msg.frontier_point.y = frontierPoint[1]
        msg.heading = self.heading
        msg.angular_velocity = self.angularVelocity

        self.driftData_pub.publish(msg)

    def publishCenterLine(self):
        if self.centerline is None:
            return
        msg = CenterLine()
        for point in self.centerline:
            msg.points.append(Point(x=point[0], y=point[1]))

        self.centerLine_pub.publish(msg)

    def lidar_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)

        angles = np.pi/2 + msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        max_range = msg.range_max

        # Filter invalid points
        valid = (ranges > 0) & (ranges < max_range*0.9)
        x_coords = ranges[valid] * np.cos(angles[valid])
        y_coords = ranges[valid] * np.sin(angles[valid])

        x_coords, y_coords = self.update_occupancy_grid(x_coords, y_coords)
        self.update_boundary(x_coords, y_coords)
        try:
            self.segments
        except:
            return

        self.centerline = self.calculate_centerline(self.segments)
        self.calculateAngularVelocity()
        self.publishDrift()
        self.publishCenterLine()

        if DEBUG:
            self.update_visualization()

    def calculateAngularVelocity(self):
        currentTime = time.time()
        if self.prevHeading is None or self.prevTime is None:
            self.prevHeading = self.heading
            self.prevTime = currentTime
            self.angularVelocity = 0

        delta_heading = self.heading - self.prevHeading
        delta_time = currentTime - self.prevTime

        delta_heading = (delta_heading + np.pi) % (2 * np.pi) - np.pi
        self.angularVelocity = delta_heading / delta_time

        self.prevHeading = self.heading
        self.prevTime = currentTime

    def update_occupancy_grid(self, x_coords, y_coords):
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.int8)
        x_cells = np.floor(x_coords / self.grid_resolution + self.grid_center[0]).astype(int)
        y_cells = np.floor(y_coords / self.grid_resolution + self.grid_center[1]).astype(int)

        x_cells = np.clip(x_cells, 0, self.grid_size[0] - 1)
        y_cells = np.clip(y_cells, 0, self.grid_size[1] - 1)
        return x_cells, y_cells

    def update_boundary(self, x_coords, y_coords):
        left_boundary, right_boundary = self.classify_boundaries(x_coords, y_coords)

        if len(left_boundary) < 20 or len(right_boundary) < 20:
            return

        left_length = np.sum(np.sqrt(np.sum(np.diff(left_boundary, axis=0)**2, axis=1)))
        right_length = np.sum(np.sqrt(np.sum(np.diff(right_boundary, axis=0)**2, axis=1)))


        if left_length > right_length:
            self.b_long, self.b_short = left_boundary, right_boundary
            self.isLeftLong = True
            self.b_long = self.b_long[::-1]
        else:
            self.b_long, self.b_short = right_boundary, left_boundary
            self.isLeftLong = False

        self.b_long = self.resample_boundary(self.b_long, resampleBoundary_long)
        self.b_short = self.resample_boundary(self.b_short, resampleBoundary_short)

        self.heading = self.guessHeading()

        self.segments = self.calculate_segments(self.b_long, self.b_short, segmentMaxLength)
        self.segments = self.filter_overlapping_segments(self.segments)

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
                    if abs(delta - avg_delta) < boundaryDeltaThreshold:
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
        segments = []
        used_short_indices = set()
        len_long = len(b_long)

        for i in range(len_long):
            distances = np.linalg.norm(b_short - b_long[i], axis=1)
            nearest_idx = np.argmin(distances)

            if nearest_idx in used_short_indices:
                continue

            if distances[nearest_idx] < w_max:
                segments.append([b_long[i], b_short[nearest_idx]])
                if b_short[nearest_idx][1] < 80:
                    used_short_indices.add(nearest_idx)
            else:
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
                    if point[1] > 25:
                        projected_point = point + normals[j] * avg_width
                        segments.append([point, projected_point])
                    
                break  # End loop after projecting the remaining section

        return segments
    
    def filter_overlapping_segments(self, segments, min_distance=5.0):
        """
        Remove segments with overlapping or very close midpoints.
        """
        filtered_segments = []
        prev_midpoint = None

        for segment in segments:
            midpoint = (segment[0] + segment[1]) / 2
            if prev_midpoint is None or np.linalg.norm(midpoint - prev_midpoint) > min_distance:
                filtered_segments.append(segment)
                prev_midpoint = midpoint

        return filtered_segments

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
                if self.isLeftLong:
                    normal = np.array([tangent[1], -tangent[0]])
                else:
                    normal = np.array([-tangent[1], tangent[0]])
            else:
                normal = np.array([0, 0])  # Default normal if tangent is degenerate

            normals.append(normal)

        # Convert normals to numpy array
        normals = np.array(normals)

        return normals

    def calculate_centerline(self, segments):
        center_points = [(segment[0] + segment[1]) / 2 for segment in segments]

        
        center_points = np.array(center_points)
        x, y = center_points[:, 0], center_points[:, 1]
        tck, _ = splprep([x, y], s=3)

        smooth_points = np.linspace(0, 1, 100)
        x_smooth, y_smooth = splev(smooth_points, tck)

        return np.column_stack((x_smooth, y_smooth))


    def guessHeading(self):
        position = np.array([100, 25])
        left_distances = np.linalg.norm(self.b_long - position, axis=1)
        right_distances = np.linalg.norm(self.b_short - position, axis=1)

        nearest_left = self.b_long[np.argmin(left_distances)]
        nearest_right = self.b_short[np.argmin(right_distances)]

        tangent = nearest_right - nearest_left

        heading =  np.arctan2(tangent[1], tangent[0])
        return heading


    def update_visualization(self):
        # Clear the axes
        self.ax.clear()

        # Plot the boundaries
        self.ax.plot(self.b_long[:, 0], self.b_long[:, 1], 'r-', label="Long Boundary")
        self.ax.plot(self.b_short[:, 0], self.b_short[:, 1], 'b-', label="Short Boundary")

        if hasattr(self, 'segments'):  # Ensure segments exists
            for segment in self.segments:
                # Each segment contains two points: [point_long, point_short]
                point_long, point_short = segment
                self.ax.plot(
                    [point_long[0], point_short[0]],
                    [point_long[1], point_short[1]],
                    'g-'  # Green line for the segment
                )
        
        if hasattr(self, 'centerline'):  # Ensure centerline exists
            self.ax.plot(
                self.centerline[:, 0],  # x-coordinates of the centerline
                self.centerline[:, 1],  # y-coordinates of the centerline
                'c-',  # Cyan line for the centerline
                label="Centerline"
            )

        # Optional: Add legend
        self.ax.legend(loc='upper left',
                       bbox_to_anchor=(1.05, 1))

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