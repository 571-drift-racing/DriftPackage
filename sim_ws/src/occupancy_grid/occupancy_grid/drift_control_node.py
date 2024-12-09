import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from racecar_interfaces.msg import DriftData


from collections import namedtuple
import numpy as np
import math

############## Constants ##############
wheelbase = .3302 #car length meters

friction_err = .1
speed_err = 1
angle_err = .3
lf = wheelbase / 2
SPINOUT  = 1000
start_drift_angle = 1.1
inital_angle_compensation = 0.1
############## Constants ##############

DEBUG = False
Velocity = namedtuple('Velocity', ['speed', 'angle'])

class DriftControlNode(Node):
    def __init__(self):
        super().__init__('drift_control_node')

        self.heading = None
        self.curSpeed = 0
        self.processingSignal = False
        self.prevSteering = 0.0
        self.prevSpeed = 0.0
        self.drift_data_sub = self.create_subscrition(
                          DriftData,
                          '/drift_data',
                          self.drift_callback,
                          10
        )

        # msg = DriftData()
        # msg.middlePoint.x = middlePoint[0]
        # msg.middlePoint.y = middlePoint[1]
        # msg.frontierPoint.x = frontierPoint[0]
        # msg.frontierPoint.y = frontierPoint[1]
        # msg.heading = self.heading
        # msg.angularVelocity = self.angularVelocity


    def calculateTurnAngle(self,middle, last): # uses trig to calculate angle
        a = 125 - middle
        b = last - middle
        angle = np.arccos(np.dot(a,b)/np.linalg.norm(a)*np.linalg.norm(b))
        return np.pi - angle

    def driftDecision(self,angle,speed,distance,friction): # return true if a drift is necessary with current
        t = distance/speed
        w = angle / t
        centripedal_accel = w * speed
        return centripedal_accel > friction - friction_err

    def isDrifting(self,velocity_speed, velocity_angle, heading):
        slip_angle = np.abs(velocity_angle-heading)
        return velocity_speed > speed_err and slip_angle > angle_err

    def driftControl(self,drift_angle,angularAcceleration,velocity_speed,velocity_angle):
        v_x = velocity_speed * np.cos(velocity_angle)
        v_y = velocity_speed * np.sin(velocity_angle)
        return np.arctan((v_y + angularAcceleration * lf)/ v_x) - drift_angle

    def driftStatus(self,middlePoint, finalPoint, velocity, angularVelocity, friction: .7, heading):
        angle = self.calculateTurnAngle(self,middlePoint, finalPoint) # Calculate the angle of the curve
        distance = np.linalg.norm(finalPoint-125)
        route_angle = np.arctan((middlePoint[1]-125)/(middlePoint[0]-125))
        t = distance/velocity.speed
        desiredAngularVelocity = angle / t
        shouldDrift = self.driftDecision(angle,velocity.speed,distance,friction)
        v_y = velocity.speed * np.sin(velocity.angle - heading)
        driveMsg = AckermannDriveStamped()
        drift_angle = velocity.angle - heading
        if self.isDrifting(velocity.speed,velocity.angle, heading): # start drift
            if shouldDrift:
                if(math.copysign(desiredAngularVelocity) == math.copysign(angularVelocity)):
                    driveMsg.drive.steering_angle = self.driftControl(drift_angle,desiredAngularVelocity, velocity.speed, velocity.angle)
                else:
                    driveMsg.drive.steering_angle = self.driftControl(drift_angle,desiredAngularVelocity-angularVelocity, velocity.speed, velocity.angle)
                driveMsg.drive.speed = SPINOUT
                self.cmdDrive_pub.publish(driveMsg) 
            else:
                driveMsg.drive.steering_angle = self.driftControl(drift_angle,-angularVelocity, velocity.speed, velocity.angle)
                driveMsg.drive.speed = v_y
                self.cmdDrive_pub.publish(driveMsg) 
        else:
            if shouldDrift:
                if angle > 0:
                    driveMsg.drive.steering_angle = start_drift_angle + route_angle * inital_angle_compensation
                else:
                    driveMsg.drive.steering_angle = -start_drift_angle + route_angle * inital_angle_compensation
                driveMsg.drive.speed = SPINOUT
                self.cmdDrive_pub.publish(driveMsg) 

    def drift_callback(self,msg:DriftData):
        mp = np.array([msg.middle_point.x,msg.middle_point.x])
        fp = np.array([msg.frontier_point.x,msg.frontier_point.x])
        self.driftStatus(mp,fp,
                         Velocity(speed=msg.speed, angle=msg.angle),
                         msg.angular_velocity,
                         msg.friction,
                         msg.heading)

def main(args=None):
    rclpy.init(args=args)
    drift_control_node = DriftControlNode()

    try:
        rclpy.spin(drift_control_node)
    except KeyboardInterrupt:
        drift_control_node.get_logger().info("Shutting down node.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()




