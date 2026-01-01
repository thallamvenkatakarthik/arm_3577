#!/usr/bin/env python3
'''
Team ID:          3577
Theme:            Krishi coBot
Author List:      D Anudeep, Karthik, Vishwa, Manikanta
Filename:         task4a_manipulation.py
Purpose:          UR5 servo-based pick & place on REAL HARDWARE
Behavior:
  - Initial -> P1 -> Initial -> anticlockwise offset -> P2 -> P3
  - P1: pick fertilizer -> place at P3
  - P2: pick bad fruits -> place at P3
  - Stops at P3
'''

import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
from threading import Thread

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import SetBool
from std_msgs.msg import String, Float32


# ---------------- Waypoints ----------------
WAYPOINTS = [
    {'position': [-0.214, -0.532, 0.557], 'orientation': [0.707, 0.028, 0.034, 0.707]},  # P1
    {'position': [-0.159,  0.501, 0.415], 'orientation': [0.029, 0.997, 0.045, 0.033]},  # P2
    {'position': [-0.806,  0.010, 0.182], 'orientation': [-0.684, 0.726, 0.05, 0.008]}   # P3
]

# ---------------- TF Frames ----------------
FERTILISER_FRAME = '3577_fertilizer_1'
BAD_FRUIT_FRAMES = ['3577_bad_fruit_3', '3577_bad_fruit_2', '3577_bad_fruit_1']

# ---------------- Motion Parameters ----------------
PRE_Z_OFFSET = 0.10
LIFT_Z = 0.20
TRASH_CLEARANCE_Z = 0.20

SAFE_Z_VEL = 0.05
FORCE_CONTACT_THRESH = 8.0
FORCE_HARD_LIMIT = 15.0

KP_LIN = 1.0
KP_ANG = 0.6
MAX_LIN = 0.10
MAX_ANG = 0.5

POS_TOL = 0.02
ORI_TOL = 0.20

BASE_FRAME = 'base_link'
EE_FRAME   = 'wrist_3_link'


class UR5ServoPickPlace(Node):

    def __init__(self):
        super().__init__('ur5_task4a_manipulation')

        self.pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Magnet
        self.magnet_client = self.create_client(SetBool, '/magnet')
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for magnet service...')

        # Force feedback
        self.force_z = 0.0
        self.create_subscription(Float32, '/net_wrench', self.force_cb, 10)

        # Sync
        self.dock_reached = False
        self.create_subscription(String, '/detection_status', self.dock_status_cb, 10)

        self.dt = 1.0 / 30.0
        self.start_time = time.time()
        self.tf_delay = 1.0

        self.initial_pose = None
        self.sequence = []
        self.current_target_index = 0
        self.waiting = False
        self.active = False

        self.create_timer(self.dt, self.update_loop)
        self.get_logger().info('✅ Task-4A manipulation node ready')

    # ---------------- Safety ----------------
    def stop(self):
        self.pub.publish(TwistStamped())
        time.sleep(0.05)

    # ---------------- Callbacks ----------------
    def force_cb(self, msg):
        self.force_z = msg.data
        if abs(self.force_z) > FORCE_HARD_LIMIT:
            self.get_logger().error(f'🚨 FORCE LIMIT EXCEEDED: {self.force_z:.2f} N')
            self.stop()
            self.magnet(False)

    def dock_status_cb(self, msg):
        if msg.data.startswith("DOCK_STATION"):
            self.dock_reached = True

    # ---------------- Magnet ----------------
    def magnet(self, state):
        req = SetBool.Request()
        req.data = state
        self.magnet_client.call_async(req)
        time.sleep(0.3)
        self.stop()

    # ---------------- Quaternion math ----------------
    def normalize_quat(self, q):
        q = np.array(q, dtype=float)
        return q / np.linalg.norm(q)

    def quat_conjugate(self, q):
        return np.array([-q[0], -q[1], -q[2], q[3]])

    def quat_multiply(self, q1, q2):
        x1,y1,z1,w1 = q1
        x2,y2,z2,w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    def quat_to_axis_angle(self, q):
        q = self.normalize_quat(q)
        angle = 2 * math.acos(np.clip(q[3], -1.0, 1.0))
        s = math.sqrt(max(1 - q[3]**2, 0))
        axis = np.array([1,0,0]) if s < 1e-6 else q[:3]/s
        return axis, angle

    # ---------------- TF helpers ----------------
    def get_pose(self, frame):
        try:
            t = self.tf_buffer.lookup_transform(
                BASE_FRAME, frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=2))
            pos = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ])
            quat = np.array([
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w
            ])
            return pos, self.normalize_quat(quat)
        except Exception:
            return None, None

    # ---------------- Servo control ----------------
    def pose_error(self, tpos, tquat, cpos, cquat):
        pos_err = tpos - cpos
        q_err = self.quat_multiply(tquat, self.quat_conjugate(cquat))
        axis, angle = self.quat_to_axis_angle(q_err)
        ori_err = axis * angle
        return pos_err, ori_err

    def publish_twist(self, pos_err, ori_err):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        cmd.twist.linear.x = float(np.clip(KP_LIN * pos_err[0], -MAX_LIN, MAX_LIN))
        cmd.twist.linear.y = float(np.clip(KP_LIN * pos_err[1], -MAX_LIN, MAX_LIN))
        cmd.twist.linear.z = float(np.clip(KP_LIN * pos_err[2], -MAX_LIN, MAX_LIN))

        cmd.twist.angular.x = float(np.clip(KP_ANG * ori_err[0], -MAX_ANG, MAX_ANG))
        cmd.twist.angular.y = float(np.clip(KP_ANG * ori_err[1], -MAX_ANG, MAX_ANG))
        cmd.twist.angular.z = float(np.clip(KP_ANG * ori_err[2], -MAX_ANG, MAX_ANG))

        self.pub.publish(cmd)

    def move_and_wait(self, tpos, tquat):
        while rclpy.ok():
            cpos, cquat = self.get_pose(EE_FRAME)
            if cpos is None:
                continue

            pos_err, ori_err = self.pose_error(tpos, tquat, cpos, cquat)

            if np.linalg.norm(pos_err) < POS_TOL and np.linalg.norm(ori_err) < ORI_TOL:
                self.stop()
                return

            self.publish_twist(pos_err, ori_err)
            time.sleep(self.dt)

    # ---------------- Pick / Place ----------------
    def pick_object(self, frame):
        pos, quat = self.get_pose(frame)
        if pos is None:
            return False

        self.move_and_wait(pos + np.array([0,0,PRE_Z_OFFSET]), quat)

        while self.force_z < FORCE_CONTACT_THRESH:
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.twist.linear.z = -SAFE_Z_VEL
            self.pub.publish(cmd)
            time.sleep(self.dt)

        self.stop()
        self.magnet(True)
        self.move_and_wait(pos + np.array([0,0,LIFT_Z]), quat)
        return True

    def place_at_p3(self):
        pos = np.array(WAYPOINTS[2]['position'])
        quat = np.array(WAYPOINTS[2]['orientation'])

        self.move_and_wait(pos + np.array([0,0,PRE_Z_OFFSET]), quat)
        self.move_and_wait(pos, quat)
        self.magnet(False)
        self.move_and_wait(pos + np.array([0,0,TRASH_CLEARANCE_Z]), quat)

    # ---------------- Sequence ----------------
    def create_motion_sequence(self):
        init = self.initial_pose
        return [
            init,
            WAYPOINTS[0],
            WAYPOINTS[1],
            WAYPOINTS[2]
        ]

    def perform_action_for_target(self, idx):
        if idx == 1:
            if self.pick_object(FERTILISER_FRAME):
                self.place_at_p3()

        elif idx == 4:
            for fruit in BAD_FRUIT_FRAMES:
                if self.pick_object(fruit):
                    self.place_at_p3()

        self.current_target_index += 1
        self.waiting = False

    # ---------------- Main loop ----------------
    def update_loop(self):
        if not self.dock_reached:
            return
        if time.time() - self.start_time < self.tf_delay:
            return

        cpos, cquat = self.get_pose(EE_FRAME)
        if cpos is None:
            return

        if self.initial_pose is None:
            self.initial_pose = {'position': cpos.tolist(), 'orientation': cquat.tolist()}
            self.sequence = self.create_motion_sequence()
            self.active = True
            return

        if self.waiting or not self.active:
            return

        target = self.sequence[self.current_target_index]
        tpos = np.array(target['position'])
        tquat = np.array(target['orientation'])

        pos_err, ori_err = self.pose_error(tpos, tquat, cpos, cquat)

        if np.linalg.norm(pos_err) < POS_TOL and np.linalg.norm(ori_err) < ORI_TOL:
            self.stop()
            self.waiting = True
            Thread(
                target=lambda: self.perform_action_for_target(self.current_target_index),
                daemon=True
            ).start()
            return

        self.publish_twist(pos_err, ori_err)


def main():
    rclpy.init()
    node = UR5ServoPickPlace()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
