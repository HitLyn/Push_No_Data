#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import rospy
import tf
import geometry_msgs.msg

import numpy as np

VELOCITY_LOOP_DURATION = 1
VELOCITY_UP_LIMIT = 0.1
TIME_STEP_DURATION = 0.3


rospy.init_node("velocity_push_target_action")
#
# def push(action):
#     pose_tool_start = get_object_tool_pose()
#     pose_tool_target = pose_tool_start + np.array([action.x, action.y])
#     target_time = rospy.Time.now() + rospy.Duration(VELOCITY_LOOP_DURATION)
#     vel_msg = geometry_msgs.msg.Vector3()
#     delta_pose_msg = geometry_msgs.msg.Vector3()
#     while(rospy.Time.now() < target_time):
#         pose_tool_now = get_object_tool_pose()
#         delta_pose = pose_tool_target - pose_tool_now
#         # print('delta_pose: ', delta_pose)
#         time_left = (target_time - rospy.Time.now()).to_sec()
#         # print('time_left: ', time_left)
#         velocity = delta_pose/time_left
#         vel_msg.x = np.clip(velocity[1], -VELOCITY_UP_LIMIT, VELOCITY_UP_LIMIT)
#         vel_msg.y = -np.clip(velocity[0], -VELOCITY_UP_LIMIT, VELOCITY_UP_LIMIT)
#         pub.publish(vel_msg)
#         delta_pose_msg.x = delta_pose[0]
#         delta_pose_msg.y = delta_pose[1]
#         delta_pose_pub.publish(delta_pose_msg)
#
#     print('action finished!')

def push(action):
    vel_msg = geometry_msgs.msg.Vector3()
    vel_msg.x = np.clip(0.5*action.y/TIME_STEP_DURATION, -VELOCITY_UP_LIMIT, VELOCITY_UP_LIMIT)
    vel_msg.y = -np.clip(0.5*action.x/TIME_STEP_DURATION, -VELOCITY_UP_LIMIT, VELOCITY_UP_LIMIT)
    pub.publish(vel_msg)


def get_object_tool_pose():
    # listener = tf.TransformListener()
    # listener.waitForTransform('table_gym', 'pushable_object_6', rospy.Time(), rospy.Duration(0.1))
    listener.waitForTransform('table_gym', 's_model_tool0', rospy.Time(), rospy.Duration(0.1))

    # (trans_object,rot_object) = listener.lookupTransform('table_gym', 'pushable_object_6', rospy.Time(0))
    (trans_tool,rot_tool) = listener.lookupTransform('table_gym', 's_model_tool0', rospy.Time(0))
    # pose_tool = None
    # while(pose_tool is None):
    #     try:
    #         (trans_object,rot_object) = listener.lookupTransform('table_gym', 'pushable_object_6', rospy.Time(0))
    #         (trans_tool,rot_tool) = listener.lookupTransform('table_gym', 's_model_tool0', rospy.Time(0))
    #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #         continue
    #
    #
    #
    # print('transformation received..', rospy.Time.now().to_sec())
    #
    # pose_object = np.concatenate([np.asarray(trans_object), np.asarray(rot_object)])
    pose_tool = np.asarray(trans_tool)[:2]

    return pose_tool


pub = rospy.Publisher("/dynamic_pushing/velocity", geometry_msgs.msg.Vector3, queue_size=10)
delta_pose_pub = rospy.Publisher("/dynamic_pushing/delta_pose", geometry_msgs.msg.Vector3, queue_size=10)
listener = tf.TransformListener()
rospy.Subscriber("/dynamic_pushing/target_action", geometry_msgs.msg.Point, push)
rospy.spin()
