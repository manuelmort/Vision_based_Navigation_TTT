#!/usr/bin/env python3
#import rospy

import rclpy 
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vision_based_navigation_ttt.msg import OpticalFlow
from vision_based_navigation_ttt.msg import TauComputation
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridgeError, CvBridge
import cv2

############################################################################################
# Initialization of the variables for setting the limits of the ROIs

# Extreme left and extreme right
x_init_el = 0
y_init_el = 0
x_end_el = 0
y_end_el = 0

x_init_er = 0
y_init_er = 0
x_end_er = 0
y_end_er = 0

# Left and right
x_init_l = 0
y_init_l = 0
x_end_l = 0
y_end_l = 0

x_init_r = 0
y_init_r = 0
x_end_r = 0
y_end_r = 0

# Centre
x_init_c = 0
y_init_c = 0
x_end_c = 0
y_end_c = 0

# Definition of the limits for the ROIs
def set_limit(img_width, img_height):
    
    ########## IMPORTANT PARAMETERS: ##########
    # Extreme left and extreme right
    global x_init_el
    global y_init_el
    global x_end_el
    global y_end_el
    x_init_el = 0
    y_init_el = 0
    x_end_el = int(3 * img_width / 12)
    y_end_el = int(7.5 * img_height / 12)

    global x_init_er
    global y_init_er
    global x_end_er
    global y_end_er
    x_init_er = int(9 * img_width / 12)
    y_init_er = 0
    x_end_er = int(img_width)
    y_end_er = int(7.5 * img_height / 12)

    # Left and right
    global x_init_l
    global y_init_l
    global x_end_l
    global y_end_l
    x_init_l = int(3 * img_width / 12)
    y_init_l = int(1 * img_height / 12)
    x_end_l = int(5 * img_width / 12)
    y_end_l = int(7 * img_height / 12)

    global x_init_r
    global y_init_r
    global x_end_r
    global y_end_r
    x_init_r = int(7 * img_width / 12)
    y_init_r = int(1 * img_height / 12)
    x_end_r = int(9 * img_width / 12)
    y_end_r = int(7 * img_height / 12)

    # Centre
    global x_init_c
    global y_init_c
    global x_end_c
    global y_end_c
    x_init_c = int(5.5 * img_width / 12)
    y_init_c = int(3 * img_height / 12)
    x_end_c = int(6.5 * img_width / 12)
    y_end_c = int(6 * img_height / 12)
    ###########################################

##############################################################################################

# Filtering procedure for the TTT values
def tau_filtering(vector):
    
    perc_TTT_val_discarded = 0.15
    jump = int(perc_TTT_val_discarded * np.size(vector))
    vector = np.sort(vector)
    vector = np.delete(vector, range(jump))
    vector = np.delete(vector, range(np.size(vector) - jump, np.size(vector)))

    return vector

#############################################################################################

# Computation of the average TTT
def tau_final_value(self, vector, cnt):

    if cnt >= self.min_TTT_number:
        mean = np.sum(vector) / cnt
    else:
        mean = -1

    return mean

###########################################################################################

# Visual representation of the ROIs with the average TTT values
def draw_image_segmentation(curr_image, tau_el, tau_er, tau_l, tau_r, tau_c):

    color_image = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2BGR)
    color_blue = [255, 0, 0]  
    color_green = [0, 255, 0]
    color_red = [0, 0, 255]
    linewidth = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Extreme left and extreme right
    cv2.rectangle(color_image, (x_init_el, y_init_el), (x_end_el, y_end_el), color_blue, linewidth)
    cv2.rectangle(color_image, (x_init_er, y_init_er), (x_end_er, y_end_er), color_blue, linewidth)
    cv2.putText(color_image, str(round(tau_el, 1)), (int((x_end_el+x_init_el)/2.5), int((y_end_el+y_init_el)/2)),
                font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(tau_er, 1)), (int((x_end_er+x_init_er) / 2.1), int((y_end_er+y_init_er) / 2)),
                font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Left and right
    cv2.rectangle(color_image, (x_init_l, y_init_l), (x_end_l, y_end_l), color_green, linewidth)
    cv2.rectangle(color_image, (x_init_r, y_init_r), (x_end_r, y_end_r), color_green, linewidth)
    cv2.putText(color_image, str(round(tau_l, 1)),
                (int((x_end_l + x_init_l) / 2.1), int((y_end_l + y_init_l) / 2)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(color_image, str(round(tau_r, 1)),
                (int((x_end_r + x_init_r) / 2.1), int((y_end_r + y_init_r) / 2)),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Centre 
    cv2.rectangle(color_image, (x_init_c, y_init_c), (x_end_c, y_end_c), color_red, linewidth)
    cv2.putText(color_image, str(round(tau_c, 1)),
                (int((x_end_c + x_init_c) / 2.1), int((y_end_c + y_init_c) / 2)),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow('ROIs Representation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROIs Representation', (600, 600))
    cv2.imshow('ROIs Representation', color_image)
    cv2.waitKey(10)

#######################################################################################################################


class TauComputationClass(Node):

    def __init__(self):
        super().__init__('tau_computation')
        ######## IMPORTANT PARAMETERS: ########
        # Minimum number of features needed to compute the average TTT for each ROI
        self.min_TTT_number = 10
        self.image_sub_name = "front/image_raw"
        #######################################

        # First time that the callback is called
        self.first_time = True
        # Initialize current image
        self.curr_image = None

        # Initialize Image acquisition
        self.bridge = CvBridge()
        # OpticalFlowData Subscriber
        #self.of_sub = rospy.Subscriber("optical_flow", OpticalFlow, self.callback_of)
        self.of_sub = self.create_subscription(OpticalFlow, "optical_flow", self.callback_of, 10)

        # Raw Image Subscriber
        #self.image_sub = rospy.Subscriber(self.image_sub_name, Image, self.callback_img)
        self.image_sub = self.create_subscription(Image, self.image_sub_name, self.callback_img, 10)        # Tau Computation message Publisher

        #self.tau_values = rospy.Publisher("tau_values", TauComputation, queue_size=10)
        self.tau_values = self.create_publisher(TauComputation, "tau_values", 10)


    # Callback for the Optical flow topic
    def callback_of(self, data):

        img_width = data.width
        img_height = data.height

        # Coordinates at the center of the image
        xc = np.floor(img_width/2)
        yc = np.floor(img_height/2)
        # Express all points coordinate with respect to center of the image
        x = data.x - xc
        y = data.y - yc

        # Definition of the five ROIs only the first time the callback is called
        if self.first_time:
            set_limit(img_width, img_height)
            self.first_time = False

        # Initialization tau computation extreme left and extreme right
        tau_right_e = np.array([])
        tau_left_e = np.array([])
        count_left_e = 0
        count_right_e = 0

        # Initialization tau computation left and right
        tau_right = np.array([])
        tau_left = np.array([])
        count_left = 0
        count_right = 0

        # Initialization tau computation centre
        tau_centre = np.array([])
        count_centre = 0

        # TTT values computation
        for i in range(len(x)):

            # Extreme left and right
            if (x[i] >= (x_init_er - xc)) and (y[i] >= (y_init_er - yc)) and (y[i] <= (y_end_er - yc)):
                tau_right_e = np.append(tau_right_e, (x[i]**2 + y[i]**2)**0.5 / (data.vx[i]**2 + data.vy[i]**2)**0.5)
                count_right_e += 1
            if (x[i] <= (x_end_el - xc)) and (y[i] >= (y_init_el - yc)) and (y[i] <= (y_end_el - yc)):
                tau_left_e = np.append(tau_left_e, (x[i]**2 + y[i]**2)**0.5 / (data.vx[i]**2 + data.vy[i]**2)**0.5)
                count_left_e += 1

            # Left and right
            if (x[i] >= (x_init_r - xc)) and (x[i] <= (x_end_r - xc)) \
                    and (y[i] >= (y_init_r - yc)) and (y[i] <= (y_end_r - yc)):
                tau_right = np.append(tau_right, (x[i]**2 + y[i]**2)**0.5 / (data.vx[i]**2 + data.vy[i]**2)**0.5)
                count_right += 1
            if (x[i] <= (x_end_l - xc)) and (x[i] >= (x_init_l - xc)) \
                    and (y[i] >= (y_init_l - yc)) and (y[i] <= (y_end_l - yc)):
                tau_left = np.append(tau_left, (x[i]**2 + y[i]**2)**0.5 / (data.vx[i]**2 + data.vy[i]**2)**0.5)
                count_left += 1

            # Centre
            if (x[i] >= (x_init_c - xc)) and (x[i] <= (x_end_c - xc)) \
                    and (y[i] >= (y_init_c - yc)) and (y[i] <= (y_end_c - yc)):
                tau_centre = np.append(tau_centre,
                                     (x[i] ** 2 + y[i] ** 2) ** 0.5 / (data.vx[i] ** 2 + data.vy[i] ** 2) ** 0.5)
                count_centre += 1

        # Filtering TTT values for each ROI
        # Extreme right and left
        tau_right_e = tau_filtering(tau_right_e)
        tau_left_e = tau_filtering(tau_left_e)
        # Right and left
        tau_right = tau_filtering(tau_right)
        tau_left = tau_filtering(tau_left)
        # Centre
        tau_centre = tau_filtering(tau_centre)
        # Extreme right and left
        final_tau_left_e = tau_final_value(self, tau_left_e, count_left_e)
        final_tau_right_e = tau_final_value(self, tau_right_e, count_right_e)
        print("Tau right Extreme: " + str(final_tau_right_e))
        print("Tau left Extreme: " + str(final_tau_left_e))
        # Right and left
        final_tau_left = tau_final_value(self, tau_left, count_left)
        final_tau_right = tau_final_value(self, tau_right, count_right)
        print("Tau right: " + str(final_tau_right))
        print("Tau left: " + str(final_tau_left))
        # Centre
        final_tau_centre = tau_final_value(self, tau_centre, count_centre)
        print("Tau centre: " + str(final_tau_centre))

        # Publish Tau values data to rostopic
        # Creation of TauValues.msg
        msg = TauComputation()
        msg.header.stamp.secs = data.header.stamp.secs
        msg.header.stamp.nsecs = data.header.stamp.nsecs

        msg.height = data.height
        msg.width = data.width

        msg.tau_el = final_tau_left_e
        msg.tau_er = final_tau_right_e
        msg.tau_l = final_tau_left
        msg.tau_r = final_tau_right
        msg.tau_c = final_tau_centre
        self.tau_values.publish(msg)

        # Draw the ROIs with their TTT values
        draw_image_segmentation(self.curr_image, final_tau_left_e, final_tau_right_e, final_tau_left, final_tau_right, final_tau_centre)

    # Callback for the image topic
    def callback_img(self, data):
        try:
            self.curr_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print(e)
            return


def main(args=None):
    rclpy.init(args=args)
    node = TauComputationClass()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
