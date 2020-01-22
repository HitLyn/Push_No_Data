#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import rospy
import tf
import geometry_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import sys
import copy

from dwa import DWA


STEP_ACTION = 0.03

def get_object_tool_pose(listener):
    # listener = tf.TransformListener()
    pose_tool = None
    while(pose_tool is None):
        try:
            (trans_object,rot_object) = listener.lookupTransform('table_gym', 'pushable_object_6', rospy.Time(0))
            (trans_tool,rot_tool) = listener.lookupTransform('table_gym', 's_model_tool0', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        pose_object = np.concatenate([np.asarray(trans_object), np.asarray(rot_object)])
        pose_tool = np.asarray(trans_tool)[:2]

    return pose_object, pose_tool


def set_goal_marker(x, y, theta):
    goal_marker = Marker()
    goal_marker.header.frame_id = "table_gym"
    goal_marker.header.stamp = rospy.get_rostime()
    # goal_marker.ns = "goal_marker"
    goal_marker.lifetime = rospy.Duration(30)
    goal_marker.id = 0
    goal_marker.type = 1
    goal_marker.pose.position.x = x
    goal_marker.pose.position.y = y
    goal_marker.pose.position.z = 0.0375
    goal_marker.pose.orientation.x = 0.0
    goal_marker.pose.orientation.y = 0.0
    goal_marker.pose.orientation.z = np.sin(theta/2)
    goal_marker.pose.orientation.w = np.cos(theta/2)
    goal_marker.scale.x = 0.192
    goal_marker.scale.y = 0.192
    goal_marker.scale.z = 0.075
    goal_marker.color.r = 1.0
    goal_marker.color.g = 0.0
    goal_marker.color.b = 0.0
    goal_marker.color.a = 1.0
    return goal_marker


def dwa_action_predict(pos_x = 0.3, pos_y = 0.7, theta = 30):
    # init
    rospy.init_node('target_action_prediction')

    # goal visualization and action list visualization
    goal_marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    x = float(pos_x)
    y = float(pos_y)
    theta = np.deg2rad(float(theta))
    goal_marker = set_goal_marker(x, y, theta)
    goal_marker_pub.publish(goal_marker)

    # get pos information from environment
    listener = tf.TransformListener()
    pose_object, pose_tool = get_object_tool_pose(listener) #np.array(7), np.array(2)

    # mppi initialization
    dwa = DWA(20, 2, STEP_ACTION)

    dwa.trajectory_set_goal(pos_x, pos_y, theta)
    # dwa.U_reset()
    dwa.trajectory_update_state(pose_object, pose_tool)

    # target action publisher
    target_action_publisher = rospy.Publisher('/dynamic_pushing/target_action', geometry_msgs.msg.Point, queue_size = 1)
    target_action_msg = geometry_msgs.msg.Point()

    computing_step = 0
    # target_action publish loop
    while not rospy.is_shutdown():
        goal_marker_pub.publish(goal_marker)

        pose_tool_ = copy.copy(pose_tool)
        time_start = rospy.get_time()
        # dwa.compute_cost(computing_step)
        target_action = dwa.compute_best_action(computing_step)
        # target_action = np.array([-0.00736, 0.0291])
        # theta = dwa.compute_noise_theta()
        ############## target_action_list visualization #############







        ##############################################################
        time_finish = rospy.get_time()
        time_taken = time_finish - time_start
        print('computing_step: ', computing_step)
        print('target action: ', target_action)
        # print('target theta: ', theta)
        print('time consumed for dwa_push computation(s): ', time_taken)

        target_action_msg.x = target_action[0]
        target_action_msg.y = target_action[1]
        target_action_publisher.publish(target_action_msg)

        pose_object, pose_tool = get_object_tool_pose(listener)
        real_action = pose_tool - pose_tool_
        # print('real action: ', real_action)

        dwa.trajectory_update_state(pose_object, pose_tool)
        dwa.trajectory_update_action(real_action)

        # dwa.U_update() # 2 lines
        dwa.cost_clear()
        computing_step += 1

if __name__ == '__main__':
    dwa_action_predict(sys.argv[1], sys.argv[2], sys.argv[3])
# def main():
#     mppi_push(0.75, 0.6, -30)
#
# main()
