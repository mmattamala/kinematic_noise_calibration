#!/usr/bin/env python
import os

# ROS stuff
import rospy
from KinematicCalibration import *

if __name__ == '__main__':
    node_name = os.path.splitext(os.path.basename(__file__))[0]
    rospy.init_node(node_name)
    rospy.loginfo('Starting [%s] node' % node_name)
    calib = KinematicCalibration(mode='calibration')
    rospy.loginfo('Shuting down [%s] node' % node_name)
