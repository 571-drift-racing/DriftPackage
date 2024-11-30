#!/usr/bin/env python3
"""
Reactive Node with Multiple PIDs for Position, Orientation, Velocity, and Throttle Control, enhanced with Kalman Filter
"""
"""
Every time you make a change in the code:
colcon build

To run the simulator:
ros2 launch f1tenth_gym_ros gym_bridge_launch.py 

To run follow the gap:
ros2 run gap_follow reactive_node.py

source /opt/ros/foxy/setup.bash
source install/setup.bash
"""
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import time

class PID:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_time = time.time()

    def __call__(self, current_value):
        current_time = time.time()
        dt = current_time - self._previous_time if self._previous_time else 0.0
        error = self.setpoint - current_value

        # Proportional term
        p = self.kp * error

        # Integral term
        self._integral += error * dt
        i = self.ki * self._integral

        # Derivative term
        derivative = (error - self._previous_error) / dt if dt > 0 else 0.0
        d = self.kd * derivative

        # Update previous values for the next iteration
        self._previous_error = error
        self._previous_time = current_time

        # Calculate the output
        output = p + i + d

        print(output)
        # Apply output limits
        lower, upper = self.output_limits
        if lower is not None:
            output = max(lower, output)
        if upper is not None:
            output = min(upper, output)

        return output

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_estimate=0.0, initial_covariance=1.0):
        self.process_variance = process_variance  # Q: Process noise covariance
        self.measurement_variance = measurement_variance  # R: Measurement noise covariance
        self.estimate = initial_estimate  # Initial state estimate
        self.covariance = initial_covariance  # Initial estimation covariance

    def update(self, measurement):
        # Prediction step
        self.covariance += self.process_variance

        # Measurement update step
        kalman_gain = self.covariance / (self.covariance + self.measurement_variance)
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.covariance *= (1 - kalman_gain)

        return self.estimate

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car with Position, Orientation, Velocity, and Throttle PID Control, enhanced with Kalman Filter
    """
    def __init__(self):
        super().__init__('reactive_node')
        
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Subscribe to LIDAR
        self.laserScan_sub = self.create_subscription(LaserScan,
                                                      lidarscan_topic,
                                                      self.lidar_callback,
                                                      10)
        # Publish to drive
        self.cmdDrive_pub = self.create_publisher(AckermannDriveStamped,
                                                  drive_topic,
                                                  10)
        
        self.curSpeed = 2
        self.curAngle = 0
        
        self.steering_pid = PID(1.0, 0.0, 0.1, setpoint=0.0, output_limits=(-0.4, 0.4))


        # Initialize PID controllers for position, orientation, velocity, and throttle
        # self.position_pid = PID(1.0, 0.0, 0.1, setpoint=0.0, output_limits=(-1.0, 1.0))
        # self.orientation_pid = PID(1.0, 0.0, 0.1, setpoint=0.0, output_limits=(-0.4, 0.4))
        # self.velocity_pid = PID(1.0, 0.0, 0.1, setpoint=3.0, output_limits=(0.0, 5.0))
        # self.throttle_pid = PID(1.0, 0.0, 0.1, setpoint=0.0, output_limits=(-1.0, 1.0))

        # # Initialize Kalman Filter for estimating the position and velocity
        # self.position_kf = KalmanFilter(process_variance=1e-2, measurement_variance=1e-1)
        # self.velocity_kf = KalmanFilter(process_variance=1e-2, measurement_variance=1e-1)

        self.get_logger().info('Starting ReactiveFollowGap Node with Multiple PID Controllers and Kalman Filters')

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array """
        maxVal = 2
        proc_ranges = np.array(ranges, dtype=np.float32)
        np.clip(proc_ranges, 0, maxVal, out=proc_ranges)

        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        split_idx = np.where(free_space_ranges == 0.0)[0]
        sranges = np.split(free_space_ranges,split_idx)
        len_sranges = np.array([len(x) for x in sranges])
        max_idx = np.argmax(len_sranges)
        if max_idx == 0:
            start_i = 0
            end_i = len_sranges[0]-1
        else:
            start_i = np.sum(len_sranges[:max_idx])
            end_i = start_i+len_sranges[max_idx]-1
        max_length_ranges = sranges[max_idx]
        return start_i, end_i, max_length_ranges

    def find_best_point(self, start_i, end_i, ranges):
        """ Return index of best point in ranges """
        idx_list = np.where(ranges == np.max(ranges))[0]
        best_idx = start_i + idx_list[round(len(idx_list) / 2)]
        return best_idx

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message """
        ranges = data.ranges
        proc_ranges = self.preprocess_lidar(ranges)

        # Find closest point to LiDAR
        closestPoint = min(range(len(ranges)), key=lambda x: ranges[x])

        # Eliminate all points inside 'bubble' (set them to zero) 
        left = max(0, closestPoint - 100)
        right = min(len(proc_ranges) - 1, closestPoint + 99)
        for i in range(left, right + 1):
            proc_ranges[i] = 0

        # Find max length gap
        maxStart, maxEnd, maxRanges = self.find_max_gap(proc_ranges)

        # Find the best point in the gap 
        bestPoint = self.find_best_point(maxStart, maxEnd, maxRanges)

        # Calculate desired steering angle using position and orientation PIDs
        desired_angle = data.angle_increment * bestPoint + data.angle_min
        print("gap follow angle: ", desired_angle)
        correctedAngle = desired_angle
        # correctedAngle = self.steering_pid(desired_angle)
        # if desired_angle < 0:
        #     correctedAngle = -correctedAngle

        self.curSpeed = 3.0 if abs(correctedAngle) < np.pi/6 else 2.0
        self.curAngle = correctedAngle

        # Publish Drive message
        driveMsg = AckermannDriveStamped()
        driveMsg.drive.steering_angle = self.curAngle
        driveMsg.drive.speed = self.curSpeed
        self.cmdDrive_pub.publish(driveMsg)

        # Logging for debugging
        self.get_logger().info(f'Steering: {self.curAngle}, Speed: {self.curSpeed}')

def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized with Multiple PID Controllers and Kalman Filters")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
