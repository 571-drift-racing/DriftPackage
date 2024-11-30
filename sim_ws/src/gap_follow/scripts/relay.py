#!/usr/bin/env python3

import rclpy
import rclpy.logging
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

class RelayNode(Node):
    
    def __init__(self):
        super().__init__('relay')
        self._cmd_ackermann_drive_relay_pub = self.create_publisher(AckermannDriveStamped, 'drive_relay', 10)
        self._cmd_ackermann_drive_sub = self.create_subscription(AckermannDriveStamped, 'drive', self.send_ackermann_drive_relay_command, 10)
        
        self.get_logger().info('relay has been started.')
        
    def send_ackermann_drive_relay_command(self, msg: AckermannDriveStamped):
        msg.drive.speed *= 3
        msg.drive.steering_angle *= 3
        self._cmd_ackermann_drive_relay_pub.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)
    node = RelayNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()