#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_based_navigation_ttt.msg import OpticalFlow
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import numpy as np

# ROI boundaries
x_init_el = y_init_el = y_end_el = 0
y_init_er = x_end_er = y_end_er = 0
x_end_l = y_end_l = y_init_l = y_end_r = 0
x_init_r = y_init_r = 0

def set_limit(img_width, img_height):
    global x_init_el, y_init_el, y_end_el
    x_init_el = 0
    y_init_el = 0
    y_end_el = int(11 * img_height / 12)

    global x_end_er, y_end_er, y_init_er
    x_end_er = img_width
    y_end_er = int(11 * img_height / 12)
    y_init_er = 0

    global x_end_l, y_end_l, y_init_l
    x_end_l = int(4 * img_width / 12)
    y_end_l = int(7 * img_height / 12)
    y_init_l = int(1 * img_height / 12)

    global x_init_r, y_init_r, y_end_r
    x_init_r = int(8 * img_width / 12)
    y_init_r = int(1 * img_height / 12)
    y_end_r = int(7 * img_height / 12)

def draw_optical_flow_field(gray_image, points_old, points_new):
    color_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    for i in range(len(points_new)):
        x0, y0 = int(points_old[i, 0]), int(points_old[i, 1])
        x1, y1 = int(points_new[i, 0]), int(points_new[i, 1])
        cv2.line(color_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.imshow('Optical Flow', color_img)
    cv2.waitKey(1)   # was 10ms — blocks far less now

class OFCalculator(Node):
    def __init__(self, param):
        super().__init__('optical_flow')
        self.image_sub_name = "/camera/image"
        self.show = int(param)
        self.bridge = CvBridge()

        self.prev_image = None
        self.prev_kps = np.array([], dtype='f')
        self.prev_time = 0.0
        self.tracking = False
        self.min_feat_threshold = 0.6

        # ---- LIGHTENED: reduced feature counts ------------------------
        self.num_ext_features = 80    # was 150
        self.num_cen_features = 30    # was 50
        # FIXED: absolute floor, not half of max — the old formula (95) exceeded
        # what low-texture walls can provide, causing an endless re-detect loop
        self.min_num_features = 25

        # Contrast enhancement — helps ORB find corners on low-texture walls
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ---- LIGHTENED: frame skipping --------------------------------
        self.frame_skip = 0     # process every frame (obstacles need fast detection)
        self.frame_cnt  = 0

        # ---- Periodic re-detection ------------------------------------
        # Refresh features every N processed frames so NEW objects entering
        # the view get features quickly (LK only tracks existing points —
        # without this, an approaching obstacle stays featureless until
        # tracking collapses entirely)
        self.redetect_every = 10   # at 15 Hz → refresh every ~0.7 s
        self.since_detect   = 0

        self.roi_el = np.array([])
        self.roi_er = np.array([])
        self.roi_c = np.array([])

        self.orb_extreme = cv2.ORB_create(self.num_ext_features)
        self.orb_center = cv2.ORB_create(self.num_cen_features)

        # ---- LIGHTENED: smaller LK window, fewer pyramid levels -------
        self.lk_params = dict(winSize=(11, 11), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.image_sub = self.create_subscription(Image, self.image_sub_name, self.callback, 1)
        # queue depth 1: drop stale frames instead of queueing them

        self.optic_flow_pub = self.create_publisher(OpticalFlow, "optical_flow", 10)

    def callback(self, data):
        # ---- LIGHTENED: skip frames -----------------------------------
        self.frame_cnt += 1
        if self.frame_skip and (self.frame_cnt % (self.frame_skip + 1) != 0):
            return

        try:
            curr_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        # Contrast enhancement for feature-poor scenes
        curr_image = self.clahe.apply(curr_image)

        secs = data.header.stamp.sec
        nsecs = data.header.stamp.nanosec
        curr_time = float(secs) + float(nsecs) * 1e-9

        if self.prev_image is None:
            self.prev_image = curr_image
            self.prev_time = curr_time
            set_limit(data.width, data.height)
            self.prev_kps = self.detect_features(curr_image)
            self.tracking = len(self.prev_kps) > 0
            return

        if not self.tracking or self.since_detect >= self.redetect_every:
            self.prev_kps = self.detect_features(curr_image)
            self.tracking = len(self.prev_kps) > 0
            self.since_detect = 0
            if not self.tracking:
                self.prev_image = curr_image
                self.prev_time = curr_time
                return

        self.since_detect += 1

        tracked_features, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_image, curr_image,
                                                               self.prev_kps, None,
                                                               **self.lk_params)
        good_kps_new = tracked_features[status == 1]
        good_kps_old = self.prev_kps[status == 1]

        if len(good_kps_new) < self.min_feat_threshold * len(self.prev_kps) or len(good_kps_new) <= self.min_num_features:
            self.tracking = False
            self.prev_kps = np.array([], dtype='f')
            return
        else:
            self.prev_kps = good_kps_new.reshape(-1, 1, 2)

        dt = curr_time - self.prev_time
        if dt <= 0:
            dt = 1e-3
        flow = good_kps_new - good_kps_old

        if self.show == 1:
            draw_optical_flow_field(curr_image, good_kps_old, good_kps_new)

        msg = OpticalFlow()
        msg.header.stamp.sec = secs
        msg.header.stamp.nanosec = nsecs
        msg.height = data.height
        msg.width = data.width
        msg.dt = dt
        msg.x = good_kps_old[:, 0]
        msg.y = good_kps_old[:, 1]
        msg.vx = flow[:, 0] / dt
        msg.vy = flow[:, 1] / dt
        self.optic_flow_pub.publish(msg)

        self.prev_image = curr_image
        self.prev_time = curr_time

    def detect_features(self, image):
        self.roi_el = image[y_init_el:y_end_el, x_init_el:x_end_l]
        self.roi_er = image[y_init_er:y_end_er, x_init_r:x_end_er]
        self.roi_c = image[y_init_l:y_end_r, x_end_l:x_init_r]

        keypoints = []

        for roi, x_off, y_off, orb in [
            (self.roi_el, x_init_el, y_init_el, self.orb_extreme),
            (self.roi_er, x_init_r, y_init_er, self.orb_extreme),
            (self.roi_c, x_end_l, y_init_l, self.orb_center)]:

            kp = orb.detect(roi)
            for k in kp:
                x, y = k.pt
                k.pt = (x + x_off, y + y_off)
            keypoints.extend(kp)

        if keypoints:
            pts = cv2.KeyPoint_convert(keypoints)
            return np.float32(pts.reshape(-1, 1, 2))

        return np.array([], dtype='f')

def main(args=None):
    rclpy.init(args=args)
    # Parse only non-ROS args; default to 0 (no display)
    param = 0
    for a in sys.argv[1:]:
        if a.isdigit():
            param = a
            break
    node = OFCalculator(param)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Parameter = 1, verbose mode")
    main()
