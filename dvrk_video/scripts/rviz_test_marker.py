#!/usr/bin/python

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
import rospy

rospy.init_node('marker_test', anonymous=True)
publisher = rospy.Publisher("marker_test", Marker, queue_size=2)
count = 0

position = Point(x=0, y=0, z=2)
orientation = Quaternion(x=0, y=0, z=0, w=1)
pose = Pose(position=position, orientation=orientation)
scale = Vector3(x=1, y=1, z=1)
color = ColorRGBA(r=0.6, g=0.4, b=0.85, a=1)

while not rospy.is_shutdown():
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.type = Marker.SPHERE
    marker.pose = pose
    marker.scale = scale
    marker.color = color

    publisher.publish(marker)

    rospy.sleep(0.01)
