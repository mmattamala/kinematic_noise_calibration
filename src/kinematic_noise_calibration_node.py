#!/usr/bin/env python

# ROS stuff
import rospy
#import rosbag
import rospkg

# Transformations
import tf
import tf.transformations as tr
from tf import TransformListener

# topic filters
import message_filters

# Sensors
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage

# Utils
import SE3UncertaintyLib as SE3Lib

# python
import os
from time import time, sleep
import threading
import numpy as np
import struct
import math
import yaml
from datetime import datetime

class NoiseFromRobot(object):
    def __init__(self):
        # Read parameters to configure the node
        self.rosbag_name = self.read_parameter('~rosbag_name', 'test.bag')

        self.topic_gt_torso = self.read_parameter('~topic_gt_torso', 'rigid_body_2')
        self.topic_gt_head  = self.read_parameter('~topic_gt_head' , 'rigid_body_1')
        self.topic_gt_lfoot = self.read_parameter('~topic_gt_lfoot', 'rigid_body_4')
        self.topic_gt_rfoot = self.read_parameter('~topic_gt_rfoot', 'rigid_body_5')

        self.topic_robot_torso = self.read_parameter('~topic_robot_torso', 'torso')
        self.topic_robot_head  = self.read_parameter('~topic_robot_head' , 'gaze')
        self.topic_robot_lfoot = self.read_parameter('~topic_robot_lfoot', 'r_sole')
        self.topic_robot_rfoot = self.read_parameter('~topic_robot_rfoot', 'r_sole')

        # rospack
        self.rospack = rospkg.RosPack()

        # read data from dataset
        self.script_dir = self.rospack.get_path('kinematic_noise_calibration')

        # lists to save transformations
        self.tf_gt_T_t_h     = []
        self.tf_gt_T_t_lf    = []
        self.tf_gt_T_t_rf    = []
        self.tf_robot_T_t_h  = []
        self.tf_robot_T_t_lf = []
        self.tf_robot_T_t_rf = []

        # prepare tf subscription
        self.transform_listener = TransformListener()
        rospy.Subscriber("/imu", Imu, self.tf_callback)

        # rospy spin
        rospy.spin()
        rospy.loginfo('Computing noises')
        self.compute_noises()


    def __del__(self):
        #rospy.loginfo('Computing noises')
        #self.compute_noises()

        # define output dir
        filename = self.script_dir + '/covariances/'
        rospy.loginfo('Saved files at %s' % filename)

    def tf_callback(self, imu_msg):
        #rospy.loginfo('implement')

        timestamp = imu_msg.header.stamp

        if self.transform_listener.frameExists(self.topic_gt_torso) and     \
            self.transform_listener.frameExists(self.topic_gt_head) and     \
            self.transform_listener.frameExists(self.topic_gt_lfoot) and    \
            self.transform_listener.frameExists(self.topic_gt_rfoot) and    \
            self.transform_listener.frameExists(self.topic_robot_torso) and \
            self.transform_listener.frameExists(self.topic_robot_head) and  \
            self.transform_listener.frameExists(self.topic_robot_lfoot) and \
            self.transform_listener.frameExists(self.topic_robot_rfoot):

            # gt read torso to head
            t = self.transform_listener.getLatestCommonTime(self.topic_gt_head, self.topic_gt_torso)
            position, quaternion = self.transform_listener.lookupTransform(self.topic_gt_head, self.topic_gt_torso, t)
            self.tf_gt_T_t_h.append(self.position_quaternion_to_matrix(position, quaternion))

            # gt read torso to left foot
            t = self.transform_listener.getLatestCommonTime(self.topic_gt_lfoot, self.topic_gt_torso)
            position, quaternion = self.transform_listener.lookupTransform(self.topic_gt_lfoot, self.topic_gt_torso, t)
            self.tf_gt_T_t_lf.append(self.position_quaternion_to_matrix(position, quaternion))

            # gt read torso to right foot
            t = self.transform_listener.getLatestCommonTime(self.topic_gt_rfoot, self.topic_gt_torso)
            position, quaternion = self.transform_listener.lookupTransform(self.topic_gt_rfoot, self.topic_gt_torso, t)
            self.tf_gt_T_t_rf.append(self.position_quaternion_to_matrix(position, quaternion))

            # robot read torso to head
            t = self.transform_listener.getLatestCommonTime(self.topic_robot_head, self.topic_robot_torso)
            position, quaternion = self.transform_listener.lookupTransform(self.topic_robot_head, self.topic_robot_torso, t)
            self.tf_robot_T_t_h.append(self.position_quaternion_to_matrix(position, quaternion))

            # robot read torso to left foot
            t = self.transform_listener.getLatestCommonTime(self.topic_robot_lfoot, self.topic_robot_torso)
            position, quaternion = self.transform_listener.lookupTransform(self.topic_robot_lfoot, self.topic_robot_torso, t)
            self.tf_robot_T_t_lf.append(self.position_quaternion_to_matrix(position, quaternion))

            # robot read torso to right foot
            t = self.transform_listener.getLatestCommonTime(self.topic_robot_rfoot, self.topic_robot_torso)
            position, quaternion = self.transform_listener.lookupTransform(self.topic_robot_rfoot, self.topic_robot_torso, t)
            self.tf_robot_T_t_rf.append(self.position_quaternion_to_matrix(position, quaternion))

    def compute_noises(self):
        rospy.loginfo('Not implemented yet')
        cov = {}

        labels = ['gt_torso_head', 'gt_torso_lfoot', 'gt_torso_rfoot', 'robot_torso_head', 'robot_torso_lfoot', 'robot_torso_rfoot']
        transformations = [self.tf_gt_T_t_h, self.tf_gt_T_t_lf, self.tf_gt_T_t_rf, self.tf_robot_T_t_h, self.tf_robot_T_t_lf, self.tf_robot_T_t_rf]

        for l,Trans in zip(labels, transformations):
            #print(l)
            N = len(Trans)
            cov_fusion = [np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])] * N
            Tmean, Tsigma = SE3Lib.Fusing(Trans, cov_fusion)
            #print('Tmean')
            #print(Tmean)

            # compute deviation
            errors = []
            for T in Trans:
                xi = SE3Lib.TranToVec(SE3Lib.TransformInv(Tmean) * T)
                errors.append(xi)

            mean_cov = np.zeros((6, 6), dtype=float)
            for xi in errors:
                mean_cov = mean_cov + np.outer(xi,xi);
            #print(errors)
            #print(mean_cov)
            mean_cov = mean_cov / len(errors)
            cov[l] = mean_cov

        for key in cov:
            print('%s:' % key )
            print(cov[key])

        now = datetime.now()
        filename = "covariances/cov_%s_%s_%s_%s_%s_%s.yaml" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        np.save(filename, cov)
        #with open(filename, 'w') as outfile:
        #    yaml.dump(cov, outfile, default_flow_style=False)

    def read_parameter(self, name, default):
        """
        Get a parameter from the ROS parameter server. If it's not found, a
        warn is printed.
        @type name: string
        @param name: Parameter name
        @param default: Default value for the parameter. The type should be
        the same as the one expected for the parameter.
        @return: The resulting parameter
        """
        if not rospy.has_param(name):
            rospy.logwarn('Parameter [%s] not found, using default: %s' % (name, default))
        return rospy.get_param(name, default)

    # the functions below are from Duckietown's software
    def pose_to_matrix(self, p):
        # Return the 4x4 homogeneous matrix for a PoseStamped.msg p from the geometry_msgs
        trans = (p.pose.position.x, p.pose.position.y, p.pose.position.z)
        rot = (p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w)
        return np.dot(tr.translation_matrix(trans), tr.quaternion_matrix(rot))

    def transform_to_matrix(self, T):
        # Return the 4x4 homogeneous matrix for a TransformStamped.msg T from the geometry_msgs
        trans = (T.translation.x, T.translation.y, T.translation.z)
        rot = (T.rotation.x, T.rotation.y, T.rotation.z, T.rotation.w)
        return np.dot(tr.translation_matrix(trans), tr.quaternion_matrix(rot))

    def position_quaternion_to_matrix(self, trans, rot):
        # Return the 4x4 homogeneous matrix from a position and a quaternion
        return np.dot(tr.translation_matrix(trans), tr.quaternion_matrix(rot))


if __name__ == '__main__':
    node_name = os.path.splitext(os.path.basename(__file__))[0]
    rospy.init_node(node_name)
    rospy.loginfo('Starting [%s] node' % node_name)
    dataset_to_ros = NoiseFromRobot()
    rospy.loginfo('Shuting down [%s] node' % node_name)
