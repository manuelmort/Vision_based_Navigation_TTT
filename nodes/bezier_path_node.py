#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import numpy as np
import cv2

def cubic_bezier(P0, P1, P2, P3, n=50):
    t = np.linspace(0, 1, n).reshape(-1, 1)
    return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3

def make_bezier_segment(P_i, flow_vec, step_size=60.0, scale=3.0):
    if np.linalg.norm(flow_vec) < 1e-6:
        return None, P_i
    D_i = flow_vec / np.linalg.norm(flow_vec)
    s_i = np.linalg.norm(flow_vec) * scale
    P_next = P_i + step_size * D_i
    P1 = P_i + (s_i / 3.0) * D_i
    P2 = P_next - (s_i / 3.0) * D_i
    P3 = P_next
    curve = cubic_bezier(P_i, P1, P2, P3)
    return curve, P_next

class BezierPathNode(Node):
    def __init__(self):
        super().__init__("bezier_path_node")
        self.subscription = self.create_subscription(Vector3, "/optical_flow", self.flow_callback, 10)
        self.curves = []
        self.P_current = np.array([320.0, 240.0])  # center
        self.h, self.w = 480, 640
        cv2.namedWindow("Bezier Path", cv2.WINDOW_NORMAL)

    def flow_callback(self, msg):
        flow_vec = np.array([msg.x, msg.y])
        curve, P_next = make_bezier_segment(self.P_current, flow_vec)
        if curve is not None:
            self.curves.append(curve)
            if len(self.curves) > 50:
                self.curves.pop(0)
            self.P_current = P_next

        display = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for c in self.curves:
            pts = c.astype(int)
            for j in range(len(pts) - 1):
                cv2.line(display, tuple(pts[j]), tuple(pts[j + 1]), (255, 0, 255), 2)
        cv2.imshow("Bezier Path", display)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = BezierPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
