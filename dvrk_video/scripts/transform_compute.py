#!/usr/bin/python

from geometry_msgs.msg import Quaternion, Vector3, Transform, TransformStamped
import rospy
import tf2_ros
import tf_conversions.posemath as pm
import numpy as np
import rospy

def frameToMatrix(frame):
    transform = frame.transform
    translation = (transform.translation.x, transform.translation.y, transform.translation.z)
    rotation = (transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
    pose = pm.fromTf((translation, rotation))
    matrix = pm.toMatrix(pose)
    return matrix

def matrixToFrame(matrix):
    pose = pm.fromMatrix(matrix)
    translation = pose.p
    frame = Transform()
    frame.translation = Vector3(translation.x(), translation.y(), translation.z())
    (x, y, z, w) = pose.M.GetQuaternion()
    frame.rotation = Quaternion(x, y, z, w)
    return frame

def join_transforms():
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)
    broadcaster = tf2_ros.TransformBroadcaster()

    arm_name = rospy.get_param("~arm")
    reference = rospy.get_param("~reference")
    camera = rospy.get_param("~camera")
    base = "{}_base".format(arm_name)

    rate = rospy.Rate(10.0)
    robot_to_camera = None

    while not rospy.is_shutdown():
        try:
            world_to_tip_frame = tf_buffer.lookup_transform(arm_name, reference, rospy.Time())
            world_to_tip = frameToMatrix(world_to_tip_frame)
            tip_to_base_frame = tf_buffer.lookup_transform(base, "{}_tool_wrist_link".format(arm_name), rospy.Time())
            tip_to_base = frameToMatrix(tip_to_base_frame)
        except tf2_ros.LookupException as e:
            rate.sleep()
            continue
        except tf2_ros.ConnectivityException as e:
            rate.sleep()
            continue

        if robot_to_camera is None:
            robot_to_camera = TransformStamped()
            robot_to_camera.child_frame_id = base
            robot_to_camera.header.frame_id = camera

            x = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            tip_to_base = np.matmul(tip_to_base, x)
            world_to_base_transform = np.matmul(tip_to_base, world_to_tip) 
            base_to_world_transform = np.linalg.inv(world_to_base_transform)
            robot_to_camera.transform = matrixToFrame(base_to_world_transform)

        robot_to_camera.header.stamp = rospy.Time.now()
        broadcaster.sendTransform(robot_to_camera)
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node("transform_compute")
    join_transforms()
