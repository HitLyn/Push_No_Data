#!/usr/bin/env python
import rospy
import math
import tf
import geometry_msgs.msg
import turtlesim.srv

import time
import numpy as np
import os


if __name__ == '__main__':
    rospy.init_node('object_position_listener')

    listener = tf.TransformListener()

    object_pose = rospy.Publisher('object_pose', geometry_msgs.msg.Pose,queue_size=1)
    tool_pose = rospy.Publisher('tool_pose', geometry_msgs.msg.Pose,queue_size=1)

    rate = rospy.Rate(20)
    robot_data = []
    object_data = []
    object_data_path = '/homeL/cong_pushing/data/trajectory_data/object'
    robot_data_path = '/homeL/cong_pushing/data/trajectory_data/robot'
    while not rospy.is_shutdown() :
        try:
            (trans_object,rot_object) = listener.lookupTransform('table_gym', 'pushable_object_6', rospy.Time(0))
            (trans_tool,rot_tool) = listener.lookupTransform('table_gym', 's_model_tool0', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # print('continue')
            continue

        pose_object = geometry_msgs.msg.Pose()
        pose_tool = geometry_msgs.msg.Pose()

        pose_object.position.x = trans_object[0]
        pose_object.position.y = trans_object[1]
        pose_object.position.z = trans_object[2]

        pose_object.orientation.x = rot_object[0]
        pose_object.orientation.y = rot_object[1]
        pose_object.orientation.z = rot_object[2]
        pose_object.orientation.w = rot_object[3]
        print(trans_object)
        # print(trans_tool)

        #
        pose_tool.position.x = trans_tool[0]
        pose_tool.position.y = trans_tool[1]
        pose_tool.position.z = trans_tool[2]

        pose_tool.orientation.x = rot_tool[0]
        pose_tool.orientation.y = rot_tool[1]
        pose_tool.orientation.z = rot_tool[2]
        pose_tool.orientation.w = rot_tool[3]

        object_pose.publish(pose_object)
        tool_pose.publish(pose_tool)

        robot_data.append(trans_tool)
        object_data.append(trans_object + rot_object)

        rate.sleep()
    file_time = str(int(time.time()))
    robot_data_np = np.asarray(robot_data)
    object_data_np = np.asarray(object_data)
    np.save(os.path.join(robot_data_path, 'robot' + file_time), robot_data_np)
    np.save(os.path.join(object_data_path, 'object' + file_time), object_data_np)
