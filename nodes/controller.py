#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from vision_based_navigation_ttt.msg import TauComputation
import numpy as np


def threshold(value, limit):
    return float(np.clip(value, -limit, limit))


class Controller(Node):
    """Pure tau-balancing corridor controller with a sense/act cycle."""

    def __init__(self):
        super().__init__('controller')

        # ------------------------------------------------------------------
        # Tunable parameters
        # ------------------------------------------------------------------
        self.velocity   = 0.8
        self.kp         = 0.6    # gain on medium ROI tau difference
        self.kp_e       = 0.6    # gain on extreme ROI tau difference
        self.max_u      = 0.8
        self.percentage = 0.05   # fraction of lowest tau samples discarded

        # Sense/Act timing (seconds) — driven by callback rate, no sleeps
        self.sense_duration = 0.08
        self.act_duration   = 0.08

        # Boot phase
        self.init_cnt = 0
        self.max_init = 20

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.sense       = True
        self.sense_start = None
        self.act_start   = None
        self.control     = 0.0

        self.taus = {'er': [], 'el': [], 'r': [], 'l': []}
        self.mean = {'er': None, 'el': None, 'r': None, 'l': None}

        # ------------------------------------------------------------------
        # Publisher / Subscriber
        # ------------------------------------------------------------------
        self.steering_signal = self.create_publisher(
            TwistStamped, 'jackal_velocity_controller/cmd_vel', 10)
        self.subscription = self.create_subscription(
            TauComputation, 'tau_values', self.callback, 10)

        self.get_logger().info('Tau-balancing controller started.')

    # ------------------------------------------------------------------
    def _trimmed_mean(self, arr):
        a = np.array(arr)
        if a.size == 0:
            return None
        trimmed = a[int(self.percentage * a.size):]
        return float(np.mean(trimmed)) if trimmed.size > 0 else None

    def _publish(self, angular):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x  = float(self.velocity)
        msg.twist.angular.z = float(threshold(angular, self.max_u))
        self.steering_signal.publish(msg)

    # ------------------------------------------------------------------
    def callback(self, data):
        now = self.get_clock().now().nanoseconds * 1e-9

        # ---- Boot phase: drive straight while OF stabilises -----------
        if self.init_cnt < self.max_init:
            self._publish(0.0)
            self.init_cnt += 1
            return

        # ---- Sense phase: accumulate tau samples ----------------------
        if self.sense:
            if self.sense_start is None:
                self.sense_start = now

            if data.tau_er >= 0: self.taus['er'].append(data.tau_er)
            if data.tau_el >= 0: self.taus['el'].append(data.tau_el)
            if data.tau_r  >= 0: self.taus['r'].append(data.tau_r)
            if data.tau_l  >= 0: self.taus['l'].append(data.tau_l)

            # Hold last control while sensing
            self._publish(self.control)

            if (now - self.sense_start) < self.sense_duration:
                return

            # End of sense: average each ROI
            for k in self.taus:
                self.mean[k] = self._trimmed_mean(self.taus[k])
                self.taus[k] = []

            self.sense_start = None
            self.sense       = False
            self.act_start   = now
            return

        # ---- Act phase: tau balancing ---------------------------------
        control_e = 0.0
        control_m = 0.0

        if self.mean['el'] is not None and self.mean['er'] is not None:
            control_e = self.mean['el'] - self.mean['er']
        if self.mean['l'] is not None and self.mean['r'] is not None:
            control_m = self.mean['l'] - self.mean['r']

        self.control = self.kp_e * control_e + self.kp * control_m
        self._publish(self.control)
        self.get_logger().info(
            f"tau_el={self.mean['el']}  tau_er={self.mean['er']}  "
            f"u={self.control:.3f}")

        if (now - self.act_start) >= self.act_duration:
            self.act_start = None
            self.sense     = True


def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
