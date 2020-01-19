#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import rospy
import tf
import geometry_msgs.msg
from visualization_msgs.msg import Marker

import numpy as np
import sys
import copy

from predictor import Predictor
from trajectory import Trajectory
from mppi import MPPI

STEP_LIMIT = 100
<<<<<<< HEAD
TIME_DURATION = 0.15
VELOCITY_SCALE = 0.15
STEP_ACTION = 0.025

def get_object_tool_pose():
    listener = tf.TransformListener()
    pose_tool = None
    while(pose_tool is None):
        try:
            (trans_object,rot_object) = listener.lookupTransform('table_gym', 'pushable_object_6', rospy.Time(0))
            (trans_tool,rot_tool) = listener.lookupTransform('table_gym', 's_model_tool0', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        pose_object = np.concatenate([np.asarray(trans_object), np.asarray(rot_object)])
        pose_tool = np.asarray(trans_tool)[:2]
=======
TIME_DURATION = 0.5

def get_object_tool_pose():
    listener = tf.TransformListener()
    try:
        (trans_object,rot_object) = listener.lookupTransform('pushable_object_0', 'table_gym', rospy.Time(0))
        (trans_tool,rot_tool) = listener.lookupTransform('tool', 'table_gym', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue
>>>>>>> 42978e8b76f0396a314e22abf765ae1c55f26b31

    # print(pose_tool)

    return pose_object, pose_tool

# def push_distance(pub, vel_msg):
#     now = rospy.Time.now()
#     target_time = now + rospy.Duration(TIME_DURATION)
#     while(rospy.Time.now() < target_time):
#         pub.publish(vel_msg)

def push_distance(pub, vel_msg, pose_tool_, action):
    action_distance = np.sqrt(np.square(action[0]) + np.square(action[1]))
    pushing_distance = 0.0
    rate = rospy.rate(10)
    while(pushing_distance < action_distance):
        pub.publish(vel_msg)
        _, pose_tool = get_object_tool_pose()
        pushing_distance = np.sqrt(np.square(pose_tool[0] - pose_tool_[0]) + np.square(pose_tool[1] - pose_tool_[1]))
        rate.sleep()


<<<<<<< HEAD
    vel_msg_0 = geometry_msgs.msg.Vector3()
    vel_msg_0.x = 0.0
    vel_msg_0.y = 0.0
    pub.publish(vel_msg_0)
# def push_distance(pub, vel_msg, pose_tool_, action):
#     action_distance = np.sqrt(np.square(action[0]) + np.square(action[1]))
#     pushing_distance = 0.0
#     rate = rospy.Rate(20.0)
#     while(pushing_distance < action_distance):
#         pub.publish(vel_msg)
#         _, pose_tool = get_object_tool_pose()
#         pushing_distance = np.sqrt(np.square(pose_tool[0] - pose_tool_[0]) + np.square(pose_tool[1] - pose_tool_[1]))
#         print('pushing distance: ', pushing_distance)
#         print('target action distance: ', action_distance)
#         # rate.sleep()



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


def mppi_push(pos_x = 0.3, pos_y = 0.7, theta = 30):
    # init
    rospy.init_node('mppi_push')
    # goal visualization
    goal_marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    x = copy.copy(float(pos_x))
    y = copy.copy(float(pos_y))
    theta = copy.copy(np.deg2rad(float(theta)))
    goal_marker = set_goal_marker(x, y, theta)
    goal_marker_pub.publish(goal_marker)

=======

def mppi_push(pos_x = 0.2, pos_y = 0.2, theta = 1):
    # init
    rospy.init_node('mppi_push')
    # tf listener
    # listener = tf.TransformListener()
>>>>>>> 42978e8b76f0396a314e22abf765ae1c55f26b31
    # get pos information from environment
    pose_object, pose_tool = get_object_tool_pose() #np.array(7), np.array(2)
    print('get pose')

    mppi = MPPI(80, 3, STEP_ACTION)

<<<<<<< HEAD
    mppi.trajectory_set_goal(pos_x, pos_y, theta)
=======
    mppi = MPPI(20, 3)
    mppi.trajectory_clear()
>>>>>>> 42978e8b76f0396a314e22abf765ae1c55f26b31
    mppi.U_reset()
    mppi.trajectory_update_state(pose_object, pose_tool)
    K = mppi.get_K() # get sample nums

    # publisher
    vel_pub = rospy.Publisher('/dynamic_pushing/velocity', geometry_msgs.msg.Vector3, queue_size = 1)
    vel_msg = geometry_msgs.msg.Vector3()

    # rollout with mppi algo
    for step in range(STEP_LIMIT):
        goal_marker_pub.publish(goal_marker)
        print('step: ', step)
        # pose_object, pose_tool = get_object_tool_pose()
        mppi.compute_cost(step)
        target_action = mppi.compute_noise_action() # 5 lines
<<<<<<< HEAD
        print('target action computed: ', target_action)
        # target_action = 0.03*np.clip(target_action, -1.0, 1.0)
        # print('target action clipped: ', target_action)
        costheta = target_action[1]/np.linalg.norm(target_action)
        sintheta = target_action[0]/np.linalg.norm(target_action)
        # vel_msg = geometry_msgs.msg.Vector3()
        vel_msg.x = VELOCITY_SCALE*costheta
        vel_msg.y = -VELOCITY_SCALE*sintheta
        # vel_msg = target_action/TIME_DURATION # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! need to be complete!!!!!!with clip
=======
        target_action = 0.03*np.clip(target_action, -1, 1)
        vel_msg = target_action/TIME_DURATION # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! need to be complete!!!!!!with clip
>>>>>>> 42978e8b76f0396a314e22abf765ae1c55f26b31
        # publish action through tool_velocity_control topic
        pose_tool_ = copy.copy(pose_tool)
        push_distance(vel_pub, vel_msg, pose_tool_, target_action)

        pose_object, pose_tool = get_object_tool_pose()
        real_action = pose_tool - pose_tool_
        print('real action: ', real_action)

        mppi.trajectory_update_action(real_action)
        mppi.trajectory_update_state(pose_object, pose_tool)

        mppi.U_update() # 2 lines
        mppi.cost_clear()


# if __name__ == '__main__':
#     mppi_push(sys.argv[1], sys.argv[2], sys.argv[3])
def main():
    mppi_push(0.75, 0.6, -30)

main()
