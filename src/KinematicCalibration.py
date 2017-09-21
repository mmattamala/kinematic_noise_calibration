#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# python
import os
from time import time, sleep
import threading
import numpy as np
import scipy.linalg
import struct
import math
import yaml
from datetime import datetime
import distutils.dir_util

# Utils
import SE3UncertaintyLib as SE3Lib

class KinematicCalibration():
    def __init__(self, mode='alignment'):

        self.mode = mode

        # numpy print options
        np.set_printoptions(precision=10, suppress=True)

        # Read parameters to configure the node
        self.topic_gt_torso = self.read_parameter('~topic_gt_torso', 'rigid_body_2')
        self.topic_gt_head  = self.read_parameter('~topic_gt_head' , 'rigid_body_1')
        self.topic_gt_lfoot = self.read_parameter('~topic_gt_lfoot', 'rigid_body_5')
        self.topic_gt_rfoot = self.read_parameter('~topic_gt_rfoot', 'rigid_body_4')

        self.topic_robot_torso = self.read_parameter('~topic_robot_torso', 'torso')
        self.topic_robot_head  = self.read_parameter('~topic_robot_head' , 'gaze')
        self.topic_robot_lfoot = self.read_parameter('~topic_robot_lfoot', 'r_sole')
        self.topic_robot_rfoot = self.read_parameter('~topic_robot_rfoot', 'r_sole')

        # rospack
        self.rospack = rospkg.RosPack()

        # read data from dataset
        self.script_dir = self.rospack.get_path('kinematic_noise_calibration')

        # create some dirs for configs
        self.align_dir = self.script_dir + '/alignment_config/'
        distutils.dir_util.mkpath(self.align_dir)
        self.covariance_dir = self.script_dir + '/covariances/'
        distutils.dir_util.mkpath(self.covariance_dir)


        # lists to save transformations
        self.tf_gt_T_t_h     = []
        self.tf_gt_T_t_lf    = []
        self.tf_gt_T_t_rf    = []
        self.tf_robot_T_t_h  = []
        self.tf_robot_T_t_lf = []
        self.tf_robot_T_t_rf = []

        # alignment matrices
        self.align_T_h = np.eye(4, dtype=float)
        self.align_T_lf = np.eye(4, dtype=float)
        self.align_T_rf = np.eye(4, dtype=float)

        # some lists with "pointers" to the previous list
        self.transformations = {'gt_torso_head': self.tf_gt_T_t_h,
                                'gt_torso_lfoot': self.tf_gt_T_t_lf,
                                'gt_torso_rfoot': self.tf_gt_T_t_rf,
                                'robot_torso_head': self.tf_robot_T_t_h,
                                'robot_torso_lfoot': self.tf_robot_T_t_lf,
                                'robot_torso_rfoot': self.tf_robot_T_t_rf}

        # prepare tf subscription
        self.transform_listener = TransformListener()
        # we use an imu topic as a clock to subscribe messages
        # i should not be necessary if rospy implements
        rospy.Subscriber("/imu", Imu, self.tf_callback)

        rospy.spin()

        # set periodic callback
        #rospy.Timer(rospy.Duration(1.0/100), self.tf_callback)

        # execute process
        if self.mode=='align' or self.mode == 'alignment':
            rospy.loginfo('Aligning frames')
            self.align_frames()
        elif self.mode=='calibration':
            rospy.loginfo('Computing covariances')
            self.read_alignment()
            self.compute_covariances()
        else:
            rospy.logerr('Selected mode \'%s\' is not valid. Options: \'alignment\' or \'calibration\'' % self.mode)

        rospy.loginfo('Done')

    def __del__(self):
        pass
        #if self.mode=='align' or self.mode == 'alignment':
        #    rospy.loginfo('Aligning frames')
        #    self.align_frames()
        #elif self.mode=='calibration':
        #    rospy.loginfo('Computing covariances')
        #    self.read_alignment()
        #    self.compute_covariances()
        #else:
        #    rospy.logerr('Selected mode \'%s\' is not valid. Options: \'alignment\' or \'calibration\'' % self.mode)

        #rospy.loginfo('Done')



    #def tf_callback(self, event):#, imu_msg):
    def tf_callback(self, imu_msg):
        """
        this method subscribes the tf data and saves the synchronized measurements
        into separate lists for further computations
        """
        #timestamp = imu_msg.header.stamp
        timestamp = rospy.get_time()

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

    def gauss_newton_alignment(self, listRef, listEst):
        """
        Gauss Newton following the method from:
        Salas, M., & Reid, I. D. (2015). Trajectory Alignment and Evaluation in SLAM : Horn's Method vs Alignment on the Manifold.
        In RSS Workshop: The Problem of Mobile Sensors.

        Note: It does not uses the covariance matrix
        Note2: The Jacobian of the cost function is approximated by the SE(3) adjoint
        """

        # gauss newton
        iterations = 100
        N = len(self.listRef)
        Vprev = 0   # previous value of the cost function

        # the optimal alignment transformation
        T = np.eye(4, dtype=float)
        Sigma = np.eye(6, dtype=float)

        for it in range(iterations):
            LHS = np.zeros(6)
            RHS = np.zeros(6)

            # align head frames
            for Ref,Est in zip(listRef, listEst):
                invRef = np.linalg.inv(Ref)

                delta_k = SE3Lib.TranToVec(np.dot(np.dot(invRef, T), Est)) # Eq. 2: Log(Qi^-1 * T * Pi)

                Jac = SE3Lib.TranAd(invRef)           # approximation of the Jacobian by the SE(3) Adjoint (Eq. 18)
                JacT = Jac.T
                LHS = LHS + np.dot(JacT, Jac)      # we do not weight by the information matrix
                RHS = RHS + np.dot(JacT, delta_k)

            # update
            delta = -np.linalg.solve(LHS,RHS)

            # apply update
            T = np.dot(SE3Lib.VecToTran(delta),T)
            Sigma = np.linalg.inv(LHS)

            # Check the cost function
            V = 0.
            for Ref,Est in zip(listRef, listEst):
              delta_k = SE3Lib.TranToVec(np.dot(np.dot(invRef, T), Est))
              V = V + np.dot(delta_k.T,delta_k)
            if abs(V - Vprev) < 1e-10:
              break
            Vprev = V

        return T, Sigma

    def align_frames(self, verbose=True):
        """
        Aligns ground truth and robots frames using alignment on the manifold
        We use the kinematic frame as the reference frame, so the alignment transformation converts the ground truth frame into the corresponding kinematic one

        """

        now = datetime.now()

        # head alignment
        Th,Sigmah = self.gauss_newton_alignment(self.tf_robot_T_t_h, self.tf_gt_T_t_h)
        np.save(self.align_dir + 'head.npy', Th)
        if verbose:
            print('Head transformation')
            print(Th)
            print(Sigmah)

        # left foot alignment
        Tlf,Sigmalf = self.gauss_newton_alignment(self.tf_robot_T_t_lf, self.tf_gt_T_t_lf)
        np.save(self.align_dir + 'lf.npy', Tlf)
        if verbose:
            print('Left foot transformation')
            print(Tlf)
            print(Sigmalf)

        # right foot alignment
        Trf,Sigmarf = self.gauss_newton_alignment(self.tf_robot_T_t_rf, self.tf_gt_T_t_rf)
        np.save(self.align_dir + 'rf.npy', Trf)
        if verbose:
            print('Right foot transformation')
            print(Trf)
            print(Sigmarf)

    def read_alignment(self):
        # read head alignment
        self.align_T_h = np.load(self.align_dir + 'head.npy')

        # read left foot alignment
        self.align_T_lf = np.load(self.align_dir + 'lf.npy')

        # read head alignment
        self.align_T_rf = np.load(self.align_dir + 'rf.npy')


    def empirical_covariance(self, Reflist, Estlist, Talign):
        """
        Computes the empirical covariance of a list of transformations

        Wang, Y., & Chirikjian, G. S. (2008). Second-order theory of error propagation on motion groups.
        Springer Tracts in Advanced Robotics, 47, 155–168.
        """

        cov = np.zeros((6, 6), dtype=float)
        for Ref, Est in zip(Reflist, Estlist):
            invRef = np.linalg.inv(Ref)

            err = SE3Lib.TranToVec(np.dot(np.dot(invRef, Talign), Est)) # error from predicted kinematics and ground truth, in kinematic frame
            cov = cov + np.outer(err,err);

        cov = cov / len(Reflist)

        return cov


    def compute_covariances(self, verbose=True):
        """
        Computes the empirical covariance for the head, left foot and right foot predictions

        """

        # head covariance
        Sigmah = self.empirical_covariance(self.tf_robot_T_t_h, self.tf_gt_T_t_h, self.align_T_h)
        #np.save(self.align_dir + 'head.npy', Th)
        if verbose:
            print('Head covariance')
            print(Sigmah)

        # left foot covariance
        Sigmalf = self.empirical_covariance(self.tf_robot_T_t_lf, self.tf_gt_T_t_lf, self.align_T_lf)
        #np.save(self.align_dir + 'lf.npy', Tlf)
        if verbose:
            print('Left foot covariance')
            print(Sigmalf)

        # right foot covariance
        Sigmarf = self.empirical_covariance(self.tf_robot_T_t_rf, self.tf_gt_T_t_rf, self.align_T_rf)
        #np.save(self.align_dir + 'rf.npy', Trf)
        if verbose:
            print('Right foot covariance')
            print(Sigmarf)


        #stds[key] = np.sqrt(np.diag(mean_cov))

        #for key in cov:
        #    print('%s:' % key )
        #    print(stds[key])

        #    now = datetime.now()
        #    filename = 'covariances/std_%s_%s_%s_%s_%s_%s_%s.yaml' % (key, now.year, now.month, now.day, now.hour, now.minute, now.second)
        #    #np.savetxt(filename, cov)
        #    with open(filename, 'w') as outfile:
        #        yaml.dump(list(stds), outfile, default_flow_style=False)

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
