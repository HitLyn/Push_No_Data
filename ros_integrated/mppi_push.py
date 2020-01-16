from __future__ import print_function
from __future__ import division

import rospy
import geometry_msgs.msg

import numpy as np
import sys

from predictor import Predictor
from trajectory import Trajectory
from mppi import MPPI

STEP_LIMIT = 100
TIME_DURATION = 0.3

def get_object_tool_pose():
    try:
        (trans_object,rot_object) = listener.lookupTransform('pushable_object_0', 'table_gym', rospy.Time(0))
        (trans_tool,rot_tool) = listener.lookupTransform('tool', 'table_gym', rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue

    pose_object = np.concatenate(trans_object, rot_object)
    pose_tool = trans_tool[:2]

    return pose_object, pose_tool

def push_distance(pub, vel_msg):
    now = rospy.Time.now()
    target_time = now + rospy.Duration(TIME_DURATION)
    while(rospy.Time.now() < target_time):
        pub.publish(vel_msg)

def mppi_push(pos_x = 0.1, pos_y = 0.1, theta = 1):
    # init
    rospy.init_node('mppi_push')
    # tf listener
    listener = tf.TransformListener()
    # get pos information from environment
    pose_object, pose_tool = get_object_tool_pose() #np.array(7), np.array(2)

    mppi = MPPI(40, 100)
    mppi.trajectory_clear()
    mppi.U_reset()
    mppi.trajectory_update_state(pose_object, pose_tool)
    K = mppi.get_K() # get sample nums

    # publisher
    vel_pub = rospy.Publisher('', msg_type, queue_size = 1)

    # rollout with mppi algo
    for step in range(STEP_LIMIT):
        print('step: ', step)
        # pose_object, pose_tool = get_object_tool_pose()
        mppi.compute_cost(step)
        target_action = mppi.compute_noise_action() # 5 lines
        target_action = 0.02*np.clip(target_action, -1, 1)
        vel_msg = target_action/TIME_DURATION # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! need to be complete!!!!!!with clip
        # publish action through tool_velocity_control topic
        pose_tool_ = copy.copy(pose_tool)
        push_distance(vel_pub, vel_msg)

        pose_object, pose_tool = get_object_tool_pose()
        real_action = pose_tool - pose_tool_

        mppi.trajectory_update_action(real_action)
        mppi.trajectory_update_state(pose_object, pose_tool)

        mppi.U_update() # 2 lines
        mppi.cost_clear()


if __name__ == '__main__':
    mppi_push(sys.argv[1], sys.argv[2], sys.argv[3])
