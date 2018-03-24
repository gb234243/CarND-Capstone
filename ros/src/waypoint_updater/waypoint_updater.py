#!/usr/bin/env python

from collections import deque
from geometry_msgs.msg import PoseStamped, Pose
import math
import rospy
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint


'''
This node will publish waypoints from the car's current position to some `x`
distance ahead.

As mentioned in the doc, you should ideally first implement a version which
does not care about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status
of traffic lights too.

Please note that our simulator also provides the exact location of traffic
lights and their current status in `/vehicle/traffic_lights` message.
You can use this message to build this node as well as to verify your TL
classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

# Constants
LOOKAHEAD_WPS = 200  # Number of waypoints we will publish.
POSE_QUEUE_SIZE = 50 # Number of previous vehicle poses to store.
VERBOSE = 1          # Turn logging on/off
MIN_UPDATE_DIST = 5  # Min. dist. (in m) that the ego vehicle must travel 
                     # before the list of next waypoints is updated

class WaypointUpdater(object):
    def __init__(self):
        """ Initialize the waypoint updater node:
            - Subscribe to relevant topics
            - Initialize members variables
            - Publish 'final_waypoints' topic
        """
        rospy.init_node('waypoint_updater')

        # Set start time of the node
        self.start_time = rospy.Time.now().to_sec()

        # Subscribe to 'current pose' topic
        # (Current ego position)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.ego_pose = PoseStamped()
        self.ego_pose_queue = deque(maxlen = POSE_QUEUE_SIZE) # Fixed length

        # Subscribe to 'base_waypoints' topic
        # (List of track waypoints; will only be send once)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.waypoints = []

        # Subscribe to 'traffic_waypoint' topic
        # (Index of waypoint closest to the next red traffic light. If the next 
        #  traffic light is not red, 'traffic_waypoint' is expected to be -1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        self.wp_traffic_light = -1

        # Subscribe to 'obstacle_waypoint' topic
        # (Index of waypoint closest to the next obstacle. If there is no
        #  obstacle ahead, 'obstacle_waypoint' is expected to be -1)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        self.wp_obstacle = -1

        # Publish waypoints ahead of the vehicle 
        # (Starting with the waypoint just ahead of the vehicle)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, 
                                                   queue_size = 1)
        self.final_waypoints = deque(maxlen = LOOKAHEAD_WPS) # Fixed length

        # Start node
        rospy.spin()

    def publish_waypoints(self):
        # TODO: Implement
        pass

    def pose_cb(self, ego_pose):
        """ Callback function for ego vehicle pose (position, orientation)  
            updates. If the ego vehicle travelled a certain distance 
            (MIN_UPDATE_DIST) since the last update, a new list of waypoints
            is published for the waypoint follower node.

            Arguments:
              ego_pose: Current ego pose
        """
        # Calculate the distance the ego vehicle travelled since the last
        # update
        dist_travelled = self.distance(self.get_position(ego_pose),
                                       self.get_position(self.ego_pose))
        if VERBOSE:
            rospy.loginfo('Ego pose: %s - dist(%.2f m)', 
                          self.get_pose_string(ego_pose), dist_travelled)

        # Update?
        if (dist_travelled < MIN_UPDATE_DIST):
            return

        # Set pose
        self.ego_pose = ego_pose
        
        # Keep a history of ego poses
        self.ego_pose_queue.append(self.ego_pose)

        # Publish waypoints update
        self.publish_waypoints()

    def waypoints_cb(self, waypoints):
        """ Receives a list of track waypoints and stores them internally
            
            Arguments:
              waypoints -- List of track waypoints
        """
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message.
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message.
        pass

    def get_position(self, obj):
        """ Returns the position of a 'PoseStamped' or 'Waypoint' object

            Arguments:
              obj -- 'PoseStamped' or 'Waypoint' object

            Return:
              Position of 'obj'
        """
        if (type(obj) is PoseStamped):
            return obj.pose.position
        elif (type(obj) is Waypoint):
            return obj.pose.pose.position
        assert 0, "Invalid object type (expected: 'PoseStamped', 'Waypoint')"

    def get_position_string(self, obj):
        """ Returns the position of a 'PoseStamped' or 'Waypoint' object as a
            string

            Arguments:
              obj -- 'PoseStamped' or 'Waypoint' object

            Return:
              Position of 'obj' as string
        """
        pos = self.get_position(obj)
        return ('pos(%.2f, %.2f, %.2f)' % (pos.x, pos.y, pos.z))

    def get_orientation(self, obj):
        """ Returns the orientation of a 'PoseStamped' or 'Waypoint' object

            Arguments:
              obj -- 'PoseStamped' or 'Waypoint' object

            Return:
              Orientation of 'obj'
        """
        if (type(obj) is PoseStamped):
            return obj.pose.orientation
        elif (type(obj) is Waypoint):
            return obj.pose.pose.orientation
        assert 0, "Invalid object type (expected: 'PoseStamped', 'Waypoint')"

    def get_orientation_string(self, obj):
        """ Returns the orientation of a 'PoseStamped' or 'Waypoint' object
            as a string

            Arguments:
              obj -- 'PoseStamped' or 'Waypoint' object

            Return:
              Orientation of 'obj' as a string
        """
        orient = self.get_orientation(obj)
        return ('orient(%.2f, %.2f, %.2f, %.2f)' % 
                (orient.x, orient.y, orient.z, orient.w))

    def get_waypoint_twist_string(self, wp):
        """ Returns the twist of a waypoint as a string

            Arguments:
              wp -- Waypoint

            Return:
              Twist of waypoint as a string
        """
        twl = wp.twist.twist.linear
        twa = wp.twist.twist.angular
        return ('twist[lin(%.2f, %.2f, %.2f), ang(%.2f, %.2f, %.2f)]' %
                (twl.x, twl.y, twl.z, twa.x, twa.y, twa.z))

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def get_waypoint_string(self, wp):
        """ Converts a waypoint to string and returns the string

            Arguments:
              wp -- Waypoint

            Return:
              Waypoint string
        """
        return ('%s - %s - %s' % (self.get_position_string(wp),
                                  self.get_orientation_string(wp),
                                  self.get_waypoint_twist_string(wp)))

    def get_pose_string(self, pose):
        """ Converts a time-stamped pose to string and returns the string

            Arguments:
              ps -- Time-stamped pose

            Return:
              Time-stamped pose string
        """
        return ('t(%.2f s) - %s - %s' % 
                (pose.header.stamp.to_sec() - self.start_time, 
                 self.get_position_string(pose),
                 self.get_orientation_string(pose)))

    def distance(self, p1, p2):
        """ Calculate the Euclidean distance between two positions ('p1', 'p2')

            Arguments:
              p1 -- Position 1 (x, y, z)
              p2 -- Position 2 (x, y, z)

            Return:
              Euclidean distance between 'p1' and 'p2'
        """
        return math.sqrt((p1.x - p2.x)**2 
                         + (p1.y - p2.y)**2 
                         + (p1.z - p2.z)**2)

    def distance_path(self, waypoints, wp1, wp2):
        dist = 0
        for i in range(wp1, wp2+1):
            dist += self.distance(waypoints[wp1].pose.pose.position, 
                                  waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
