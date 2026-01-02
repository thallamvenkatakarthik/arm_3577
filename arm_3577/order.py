#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray
import math


class WaypointMission(Node):
    def __init__(self):
        super().__init__('ebot_waypoint_mission')

        # ---------------- Publishers ----------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ---------------- Subscribers ----------------
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.imu_sub = self.create_subscription(Float32, '/orientation', self.imu_cb, 20)
        self.ultra_sub = self.create_subscription(
            Float32MultiArray,
            '/ultrasonic_sensor_std_float',
            self.ultra_callback,
            10
        )

        # ---------------- Waypoints (ORDERED) ----------------
        self.waypoints = [
            (-0.0034,-1.4327)
            (1.854,  -1.861),   # P1
            (4.7472, -1.7585),  # P3
            (4.7472, -0.194),   # P5
            (2.413,  -0.194),   # P6
            (1.3566,  0.0695),  # P7
            (0.5675,  1.578),   # P8
            (2.0193,  1.6684),  # P9
            (4.0595,  1.8066),  # P10
        ]

        # ---------------- State ----------------
        self.index = 0
        self.pose = None
        self.imu_yaw = None

        self.ultra_left = float('inf')
        self.ultra_right = float('inf')

        # ---------------- Parameters ----------------
        self.linear_speed = 0.5
        self.angular_speed = 0.6
        self.dist_tol = 0.15
        self.yaw_tol = math.radians(5)

        self.ultra_stop_dist = 0.25  # meters (rear safety)

        self.get_logger().info("✅ Waypoint mission with IMU + Ultrasonic started")

    # ---------------- ODOM ----------------
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.pose = (p.x, p.y)
        self.run()

    # ---------------- IMU ----------------
    def imu_cb(self, msg):
        # Convert 0–6.28 → -π to +π
        yaw = msg.data
        if yaw > math.pi:
            yaw -= 2 * math.pi
        self.imu_yaw = yaw

    # ---------------- ULTRASONIC ----------------
    def ultra_callback(self, msg):
        self.ultra_left = msg.data[4]
        self.ultra_right = msg.data[5]

    # ---------------- MAIN LOGIC ----------------
    def run(self):
        if self.pose is None or self.imu_yaw is None:
            return

        if self.index >= len(self.waypoints):
            self.cmd_pub.publish(Twist())
            return

        # -------- Rear safety check --------
        if self.ultra_left < self.ultra_stop_dist or self.ultra_right < self.ultra_stop_dist:
            self.get_logger().warn("🛑 Rear obstacle detected — emergency stop")
            self.cmd_pub.publish(Twist())
            return

        x, y = self.pose
        yaw = self.imu_yaw
        gx, gy = self.waypoints[self.index]

        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        target_yaw = math.atan2(dy, dx)
        yaw_err = self.normalize(target_yaw - yaw)

        cmd = Twist()

        # -------- Rotate first --------
        if abs(yaw_err) > self.yaw_tol:
            cmd.angular.z = math.copysign(self.angular_speed, yaw_err)
        else:
            cmd.linear.x = self.linear_speed

        # -------- Waypoint reached --------
        if dist < self.dist_tol:
            self.get_logger().info(f"📍 Reached waypoint {self.index + 1}")
            self.index += 1
            cmd = Twist()

        self.cmd_pub.publish(cmd)

    # ---------------- Helpers ----------------
    def normalize(self, ang):
        while ang > math.pi:
            ang -= 2 * math.pi
        while ang < -math.pi:
            ang += 2 * math.pi
        return ang


def main(args=None):
    rclpy.init(args=args)
    node = WaypointMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
