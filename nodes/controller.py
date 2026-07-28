#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from vision_based_navigation_ttt.msg import TauComputation
import numpy as np


def threshold(value, limit):
    return float(np.clip(value, -limit, limit))


class Controller(Node):

    def __init__(self):
        super().__init__('controller')

        # ------------------------------------------------------------------
        # Tunable parameters
        # ------------------------------------------------------------------
        self.velocity        = 1.0
        self.kp              = 0.9
        self.kp_e            = 0.9
        self.kd              = 0.5
        self.max_u           = 1.0
        self.max_control_diff = 1.0
        self.percentage      = 0.25
        self.safe_dist       = 0.5
        self.constant_left   = 1.0
        self.constant_right  = 1.0
        self.time_to_turn    = 3.0
        self.time_to_obstacle = 2.0

        # ------------------------------------------------------------------
        # Sense/Act cycle parameters — no sleeps, driven by callback rate
        # ------------------------------------------------------------------
        self.sense_duration  = 0.15   # ~2 callbacks at 15 Hz to accumulate
        self.act_duration    = 0.35   # ~5 callbacks at 15 Hz to actually turn

        # ------------------------------------------------------------------
        # Boot phase
        # ------------------------------------------------------------------
        self.init_cnt  = 0
        self.max_init  = 20

        # ------------------------------------------------------------------
        # Sense/Act state
        # ------------------------------------------------------------------
        self.sense       = True
        self.act         = False
        self.sense_start = None
        self.act_start   = None
        self.first_sense = True

        # ------------------------------------------------------------------
        # Accumulated tau arrays (sense phase)
        # ------------------------------------------------------------------
        self.final_right_e      = []
        self.final_left_e       = []
        self.final_right        = []
        self.final_left         = []
        self.tau_center_values  = []

        # ------------------------------------------------------------------
        # Averaged tau values (computed at end of sense phase)
        # ------------------------------------------------------------------
        self.mean_tau_er     = 0.0
        self.mean_tau_el     = 0.0
        self.mean_tau_r      = 0.0
        self.mean_tau_l      = 0.0
        self.mean_tau_center = 0.0

        # ------------------------------------------------------------------
        # Wall/obstacle detection flags
        # ------------------------------------------------------------------
        self.extreme_right = True
        self.extreme_left  = True
        self.right         = True
        self.left          = True
        self.center        = True
        self.obstacle      = False

        # ------------------------------------------------------------------
        # Control state
        # ------------------------------------------------------------------
        self.tau_diff           = 0.0
        self.tau_diff_extreme   = 0.0
        self.diff_left          = 0.0
        self.diff_right         = 0.0
        self.prev_diff_r        = 0.0
        self.prev_diff_l        = 0.0
        self.curr_diff_r        = 0.0
        self.curr_diff_l        = 0.0
        self.dist_from_wall_er  = 0.0
        self.dist_from_wall_el  = 0.0
        self.dist_from_wall_r   = 0.0
        self.dist_from_wall_l   = 0.0
        self.actual_wall_dist   = 1.0
        self.actual_wall_dist_e = 1.0
        self.control            = 0.0
        self.prev_controls      = []
        self.first_tdm_r        = True
        self.first_tdm_l        = True
        self.double_act_action  = False

        # ------------------------------------------------------------------
        # Publisher / Subscriber
        # ------------------------------------------------------------------
        self.steering_signal = self.create_publisher(
            TwistStamped, 'jackal_velocity_controller/cmd_vel', 10)

        self.subscription = self.create_subscription(
            TauComputation, 'tau_values', self.callback, 10)

        self.get_logger().info('Controller node started.')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _trimmed_mean(self, arr):
        """Return mean of the top (1-percentage) values, or None if empty."""
        a = np.array(arr)
        if a.size == 0:
            return None
        trimmed = a[int(self.percentage * a.size):]
        return float(np.mean(trimmed)) if trimmed.size > 0 else None

    def _publish(self, linear, angular):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x  = float(linear)
        msg.twist.angular.z = float(threshold(angular, self.max_u))
        self.steering_signal.publish(msg)

    def _find_obstacle(self, mean, t):
        self.obstacle = self.center and (mean <= t)

    def _perceive(self):
        if self.right and self.left:
            self.tau_diff = self.mean_tau_l - self.mean_tau_r
        if self.extreme_left and self.extreme_right:
            self.tau_diff_extreme = self.mean_tau_el - self.mean_tau_er

    # ------------------------------------------------------------------
    # Main callback — driven by tau_values topic rate, no sleeps
    # ------------------------------------------------------------------
    def callback(self, data):
        now = self.get_clock().now().nanoseconds * 1e-9

        # ---- Boot phase -----------------------------------------------
        if self.init_cnt < self.max_init:
            self._publish(self.velocity, 0.0)
            self.init_cnt += 1
            return

        # ---- Sense phase ----------------------------------------------
        if self.sense:
            if self.sense_start is None:
                self.sense_start = now

            # Accumulate data
            if data.tau_er >= 0: self.final_right_e.append(data.tau_er)
            if data.tau_el >= 0: self.final_left_e.append(data.tau_el)
            if data.tau_r  >= 0: self.final_right.append(data.tau_r)
            if data.tau_l  >= 0: self.final_left.append(data.tau_l)
            if data.tau_c  >= 0: self.tau_center_values.append(data.tau_c)

            # Hold last control during sense (don't cancel an ongoing turn)
            self._publish(self.velocity, self.control)

            # Check if sense duration has elapsed
            if (now - self.sense_start) < self.sense_duration:
                return

            # --- End of sense phase: compute averages ------------------
            m = self._trimmed_mean(self.final_right_e)
            if m is not None: self.mean_tau_er = m
            else:             self.extreme_right = False

            m = self._trimmed_mean(self.final_left_e)
            if m is not None: self.mean_tau_el = m
            else:             self.extreme_left = False

            m = self._trimmed_mean(self.final_right)
            if m is not None: self.mean_tau_r = m
            else:             self.right = False

            m = self._trimmed_mean(self.final_left)
            if m is not None: self.mean_tau_l = m
            else:             self.left = False

            m = self._trimmed_mean(self.tau_center_values)
            if m is not None: self.mean_tau_center = m
            else:             self.center = False

            self._perceive()

            # Update diff tracking
            if self.first_sense:
                self.prev_diff_r = self.diff_right
                self.prev_diff_l = self.diff_left
                self.curr_diff_r = self.prev_diff_r
                self.curr_diff_l = self.prev_diff_l
                self.first_sense = False
            else:
                self.prev_diff_r = self.curr_diff_r
                self.prev_diff_l = self.curr_diff_l
                self.curr_diff_r = self.diff_right
                self.curr_diff_l = self.diff_left

            # Store distances for single-wall law
            self.dist_from_wall_er = self.mean_tau_er
            self.dist_from_wall_el = self.mean_tau_el
            self.dist_from_wall_r  = self.mean_tau_r
            self.dist_from_wall_l  = self.mean_tau_l

            # Reset accumulators
            self.final_right_e     = []
            self.final_left_e      = []
            self.final_right       = []
            self.final_left        = []
            self.tau_center_values = []
            self.sense_start       = None

            # Transition to act
            self.double_act_action = True
            self.sense = False
            self.act   = True
            self.act_start = now
            return

        # ---- Act phase ------------------------------------------------
        if self.act:
            if self.act_start is None:
                self.act_start = now

            self._find_obstacle(self.mean_tau_center, self.time_to_turn)

            kp   = self.kp
            kp_e = self.kp_e
            kd   = self.kd

            control_e    = 0.0
            control_m    = 0.0
            control      = 0.0
            tau_diff_max = True

            if self.extreme_left and self.extreme_right:
                control_e    = self.tau_diff_extreme
                tau_diff_max = False
            if self.left and self.right:
                control_m    = self.tau_diff
                tau_diff_max = False

            if not tau_diff_max:
                # ---- Tau balancing (both walls visible) ---------------
                if self.obstacle:
                    self._find_obstacle(self.mean_tau_center, self.time_to_obstacle)
                    if self.extreme_left and self.extreme_right:
                        if self.obstacle and abs(self.tau_diff_extreme) < 0.5:
                            kp_e = 1.4
                            if self.double_act_action:
                                self.double_act_action = False
                                self.act_duration *= 3
                            if self.mean_tau_el > self.mean_tau_er:
                                control = kp_e * (self.mean_tau_el - self.constant_left)
                                self.get_logger().info('Obstacle! Go left')
                            else:
                                control = -kp_e * (self.mean_tau_er - self.constant_right)
                                self.get_logger().info('Obstacle! Go right')
                        else:
                            kp_e = 1.3
                            control = kp_e * control_e + kp * control_m
                            self.get_logger().info('Turn ahead')
                    else:
                        if self.obstacle and abs(self.tau_diff) < 0.5:
                            kp_e = 1.2
                            if self.double_act_action:
                                self.double_act_action = False
                                self.act_duration *= 3
                            if self.mean_tau_l > self.mean_tau_r:
                                control = kp_e * (self.mean_tau_l - self.constant_left)
                                self.get_logger().info('Obstacle! Go left (medium)')
                            else:
                                control = -kp_e * (self.mean_tau_r - self.constant_right)
                                self.get_logger().info('Obstacle! Go right (medium)')
                        else:
                            kp = 1.5
                            control = kp_e * control_e + kp * control_m
                            self.get_logger().info('Turn ahead (medium)')
                else:
                    # No obstacle — pure tau balancing with derivative
                    if len(self.prev_controls) == 2:
                        control_diff = self.prev_controls[1] - self.prev_controls[0]
                        u_diff = threshold(kd * control_diff, self.max_control_diff)
                    else:
                        u_diff = 0.0
                    u_prop  = kp_e * control_e + kp * control_m
                    control = u_prop + u_diff if u_diff * u_prop <= 0 else u_prop
                    self.get_logger().info(f'Tau Balancing  u={control:.3f}')

                self.control     = control
                self.first_tdm_r = True
                self.first_tdm_l = True

            elif self.extreme_right:
                # ---- Single wall: extreme right -----------------------
                self.get_logger().info('Single wall: extreme right')
                if self.first_tdm_r:
                    self.first_tdm_r        = False
                    self.first_tdm_l        = True
                    self.actual_wall_dist_e = 1.0
                    self.actual_wall_dist   = 1.0
                control = -kp * (self.mean_tau_er - self.actual_wall_dist_e)

            elif self.right:
                # ---- Single wall: right -------------------------------
                self.get_logger().info('Single wall: right')
                if self.first_tdm_r:
                    self.first_tdm_r      = False
                    self.first_tdm_l      = True
                    self.actual_wall_dist = 1.0
                    self.actual_wall_dist_e = 1.0
                control = -kp * (self.mean_tau_r - self.actual_wall_dist)

            elif self.extreme_left:
                # ---- Single wall: extreme left ------------------------
                self.get_logger().info('Single wall: extreme left')
                if self.first_tdm_l:
                    self.first_tdm_l        = False
                    self.first_tdm_r        = True
                    self.actual_wall_dist_e = 1.0
                    self.actual_wall_dist   = 1.0
                control = kp * (self.mean_tau_el - self.actual_wall_dist_e)

            elif self.left:
                # ---- Single wall: left --------------------------------
                self.get_logger().info('Single wall: left')
                if self.first_tdm_l:
                    self.first_tdm_l      = False
                    self.first_tdm_r      = True
                    self.actual_wall_dist = 1.0
                    self.actual_wall_dist_e = 1.0
                control = kp * (self.mean_tau_l - self.actual_wall_dist)

            self._publish(self.velocity, control)
            self.get_logger().info(f'control={control:.3f}')

            # Check if act duration has elapsed
            if (now - self.act_start) >= self.act_duration:
                # Update previous controls ring buffer
                if tau_diff_max:
                    self.prev_controls = []
                else:
                    self.prev_controls.append(self.control)
                    if len(self.prev_controls) > 2:
                        self.prev_controls.pop(0)

                # Reset for next sense phase
                self.obstacle      = False
                self.act           = False
                self.sense         = True
                self.act_start     = None
                self.extreme_left  = True
                self.extreme_right = True
                self.left          = True
                self.right         = True
                self.center        = True


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
