#!/usr/bin/env python3

import rclpy
import rclpy.logging
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32

class TalkerNode(Node):
    
    def __init__(self):
        super().__init__("talker")
        
        self.declare_parameter('v')
        self.declare_parameter('d')
        
        self._cmd_ackermann_drive_pub = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self._v = self.get_parameter('v').get_parameter_value().double_value
        self._d = self.get_parameter('d').get_parameter_value().double_value
        self._v_subscriber = self.create_subscription(Float32, 'v', self.v_callback, 10)
        self._d_subscriber = self.create_subscription(Float32, 'd', self.d_callback, 10)
        self.create_timer(0, self.send_ackermann_drive_command)
        
        self.get_logger().info("talker has been started.")
        
    def send_ackermann_drive_command(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = self._v
        msg.drive.steering_angle = self._d
        self._cmd_ackermann_drive_pub.publish(msg)
        
    def v_callback(self, msg: Float32):
        self._v = msg.data
        
    def d_callback(self, msg: Float32):
        self._d = msg.data

def main(args=None):
    rclpy.init(args=args)
    node = TalkerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()