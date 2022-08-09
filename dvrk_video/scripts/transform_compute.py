#!/usr/bin/python

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, TransformStamped
from std_msgs.msg import ColorRGBA
import rospy
import tf2_ros
import tf_conversions.posemath as pm
import PyKDL
import numpy as np

rospy.init_node('transform_compute', anonymous=True)
tf_buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tf_buffer)
broadcaster = tf2_ros.TransformBroadcaster()

rate = rospy.Rate(10.0)
camera_to_base = None
while not rospy.is_shutdown():
    try:
        camera_to_tip = tf_buffer.lookup_transform("PSM3", "camera", rospy.Time()).transform
        t, r = camera_to_tip.translation, camera_to_tip.rotation
        camera_to_tip = pm.toMatrix(pm.fromTf(((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))))
        tip_to_base = tf_buffer.lookup_transform("PSM3_base", "PSM3_tool_wrist_link", rospy.Time()).transform
        t, r = tip_to_base.translation, tip_to_base.rotation
        tip_to_base = pm.toMatrix(pm.fromTf(((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))))
    except tf2_ros.LookupException as e:
        rate.sleep()
        continue
    except tf2_ros.ConnectivityException as e:
        rate.sleep()
        continue

    if camera_to_base is None:
        camera_to_base = TransformStamped()
        camera_to_base.header.frame_id = "PSM3_base"
        camera_to_base.child_frame_id = "camera"

        x = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        tip_to_base = np.matmul(tip_to_base, x)
        camera_to_base_transform = np.matmul(tip_to_base, camera_to_tip) 
        camera_to_base_frame = pm.fromMatrix(camera_to_base_transform)
        vec = camera_to_base_frame.p
        camera_to_base.transform.translation.x = vec.x()
        camera_to_base.transform.translation.y = vec.y()
        camera_to_base.transform.translation.z = vec.z()
        quat = camera_to_base_frame.M.GetQuaternion()
        camera_to_base.transform.rotation.x = quat[0]
        camera_to_base.transform.rotation.y = quat[1]
        camera_to_base.transform.rotation.z = quat[2]
        camera_to_base.transform.rotation.w = quat[3]

    camera_to_base.header.stamp = rospy.Time.now()
    broadcaster.sendTransform(camera_to_base)
    rate.sleep()

