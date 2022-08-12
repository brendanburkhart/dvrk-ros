#!/usr/bin/python

import numpy as np
import rospy
import tf2_ros
import tf_conversions.posemath as pm
from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
    Vector3,
)
from scipy.spatial.transform import Rotation


class ReferenceFrameBroadcaster:
    """
    Computes reference frame transform for given arm, and applies it between
    subject and target frames.
    """

    def __init__(self, arm_name, reference_frame, subject_frame, target_frame):
        self.arm_name = arm_name
        self.reference_frame = reference_frame
        self.subject_frame = subject_frame
        self.target_frame = target_frame

        self.pose = None
        self.local_pose = None

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

    @staticmethod
    def _pose_to_matrix(pose: Pose):
        t, r = pose.position, pose.orientation
        translation = np.array([t.x, t.y, t.z])
        rotation = Rotation.from_quat([r.x, r.y, r.z, r.w])

        transformation = np.eye(4)
        transformation[0:3, 0:3] = rotation.as_matrix()
        transformation[0:3, 3] = translation

        return transformation

    @staticmethod
    def _matrix_to_frame(transformation):
        pose = pm.fromMatrix(transformation)

        translation = Vector3(pose.p.x(), pose.p.y(), pose.p.z())
        rotation = Quaternion(*pose.M.GetQuaternion())

        return Transform(translation, rotation)

    def measured_cp_callback(self, pose: PoseStamped):
        self.pose = self._pose_to_matrix(pose.pose)

    def local_measured_cp_callback(self, pose: PoseStamped):
        self.local_pose = self._pose_to_matrix(pose.pose)

    def broadcast(self):
        if self.pose is None or self.local_pose is None:
            return

        reference_transform = np.matmul(self.pose, np.linalg.inv(self.local_pose))
        reference_frame = self._matrix_to_frame(reference_transform)

        transform = TransformStamped()
        transform.header.frame_id = self.subject_frame
        transform.child_frame_id = self.target_frame
        transform.transform = reference_frame
        transform.header.stamp = rospy.Time.now()

        self.tf_broadcaster.sendTransform(transform)

    def run(self, rate=None):
        if rate is None:
            rate = rospy.Rate(10.0)

        self.measured_cp_sub = rospy.Subscriber(
            "/{}/measured_cp".format(self.arm_name),
            PoseStamped,
            self.measured_cp_callback,
        )
        self.local_measured_cp_sub = rospy.Subscriber(
            "/{}/local/measured_cp".format(self.arm_name),
            PoseStamped,
            self.local_measured_cp_callback,
        )

        while not rospy.is_shutdown():
            self.broadcast()
            rate.sleep()


def main():
    rospy.init_node("compute_reference_frame")

    arm_name = rospy.get_param("~arm")
    reference = rospy.get_param("~reference")
    subject = rospy.get_param("~subject")
    target = rospy.get_param("~target")

    broadcaster = ReferenceFrameBroadcaster(arm_name, reference, subject, target)
    broadcaster.run()


if __name__ == "__main__":
    main()
