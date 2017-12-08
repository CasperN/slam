#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from duckietown_msgs.msg import SegmentList, Segment, Pixel, LanePose, BoolStamped, Twist2DStamped
from apriltags_ros.msg import AprilTagDetectionArray
from duckietown_utils.instantiate_utils import instantiate
from math import sqrt, acos

class SLAM_Node(object):
    def __init__(self):
        self.node_name = "SLAM Node"
        self.active = True
        self.filter = None
        self.updateParams(None)

        self.t_last_update = rospy.get_time()
        self.velocity = Twist2DStamped()

        # Subscribers
        #self.sub = rospy.Subscriber("~segment_list", SegmentList, self.processSegments, queue_size=1)
        self.sub_switch = rospy.Subscriber("~switch", BoolStamped, self.cbSwitch, queue_size=1)
        self.sub_velocity = rospy.Subscriber("~velocity", Twist2DStamped, self.updateVelocity)

        # Publishers
        #self.pub_in_lane    = rospy.Publisher("~in_lane",BoolStamped, queue_size=1)

        # timer for updating the params
        self.timer = rospy.Timer(rospy.Duration.from_sec(1.0), self.updateParams)

        #april tags subscription
        self.sub_tags = rospy.Subscriber("~apriltag", AprilTagDetectionArray, queue_size = 1)

#apriltag functions
    def get_tag_position(self, detection):
        x = detection.pose.position.x
        y = detection.post.position.y

        r = sqrt(x**2 + y**2)
        theta = acos(x/y)
        return r,theta


# old functions from lane_filter
    def updateParams(self, event):
        if self.filter is None:
            c = rospy.get_param('~filter')
            assert isinstance(c, list) and len(c) == 2, c

            self.loginfo('new filter config: %s' % str(c))
            self.filter = instantiate(c[0], c[1])


    def cbSwitch(self, switch_msg):
        self.active = switch_msg.data


    def updateVelocity(self,twist_msg):
        self.velocity = twist_msg

    def onShutdown(self):
        rospy.loginfo("[SLAM_Node] Shutdown.")


    def loginfo(self, s):
        rospy.loginfo('[%s] %s' % (self.node_name, s))


if __name__ == '__main__':
    rospy.init_node('SLAM_Node',anonymous=False)
    lane_filter_node = SLAM_Node()
    rospy.on_shutdown(lane_filter_node.onShutdown)
    rospy.spin()
