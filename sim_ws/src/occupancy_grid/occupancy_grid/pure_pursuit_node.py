import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from racecar_interfaces.msg import CenterLine
import numpy as np

# Constants
WHEELBASE = 0.3302  # Car length in meters


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # State variables
        self.centerline = None
        self.position = np.array([100, 25])  # Current position (fixed for now)
        self.heading = 0.0  # Current heading (in radians)
        self.LOOKAHEAD_DISTANCE = 10.0  # Default lookahead distance in meters

        # Subscribers
        self.centerline_sub = self.create_subscription(
            CenterLine,
            '/center_line',
            self.centerline_callback,
            10
        )

        # Publishers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        self.get_logger().info("Pure Pursuit Node Initialized")

    def centerline_callback(self, msg):
        """
        Callback to receive the centerline points.
        """
        self.centerline = np.array([[p.x, p.y] for p in msg.points])
        if self.centerline is not None:
            self.execute_pure_pursuit()

    def execute_pure_pursuit(self):
        """
        Perform pure pursuit based on the received centerline.
        """
        if self.centerline is None or len(self.centerline) < 2:
            self.get_logger().warn("Insufficient centerline data; skipping pure pursuit!")
            return

        # Calculate the target point
        distances = np.linalg.norm(self.centerline - self.position, axis=1)
        target_idx = np.argmin(np.abs(distances - self.LOOKAHEAD_DISTANCE))

        if target_idx < 0 or target_idx >= len(self.centerline):
            self.get_logger().warn("No valid target point found; skipping pure pursuit!")
            return

        target_point = self.centerline[target_idx]

        # Calculate steering angle
        dx, dy = target_point - self.position
        alpha = np.arctan2(dy, dx) - self.heading

        # Normalize alpha to be within [-pi, pi]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        steering_angle = np.arctan(WHEELBASE * np.sin(alpha) / self.LOOKAHEAD_DISTANCE)
        speed = self.calculate_speed(steering_angle)
        self.LOOKAHEAD_DISTANCE = 10 + 2 * speed

        # Publish the drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

        self.get_logger().info(f"Steering Angle: {steering_angle:.3f}, Speed: {drive_msg.drive.speed:.2f}")

    def calculate_speed(self, steering_angle):
        """
        Calculate speed based on the steering angle.
        """
        abs_steering_angle = abs(steering_angle)
        if abs_steering_angle < np.pi / 36:
            return 8.0  # High speed for straight paths
        elif abs_steering_angle < np.pi / 20:
            return 5.0
        elif abs_steering_angle < np.pi / 18:
            return 4.0
        else:
            return 2.0  # Low speed for sharp turns


def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_node = PurePursuitNode()

    try:
        rclpy.spin(pure_pursuit_node)
    except KeyboardInterrupt:
        pure_pursuit_node.get_logger().info("Shutting down node.")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()