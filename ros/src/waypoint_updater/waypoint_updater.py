#!/usr/bin/env python

from collections import deque
from geometry_msgs.msg import Pose, PoseStamped, TwistStamped
import math
import rospy
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
import sys
import tf

'''
This node will publish waypoints from the car's current position to some `x`
distance ahead.
'''

# Constants
LOOKAHEAD_WPS = 200  # Number of waypoints we will publish.
POSE_QUEUE_SIZE = 50 # Number of previous vehicle poses to store.
VERBOSE = 1          # Turn logging on/off
MIN_UPDATE_DIST = 0.01 # Min. dist. (in m) that the ego vehicle must travel 
                       # before the list of next waypoints is updated
WP_UNDEFINED = -1    # Undefined waypoint index

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
        self.waypoint_velocities = []

        # Subscribe to 'traffic_waypoint' topic
        # (Index of waypoint closest to the next red traffic light. If the next 
        #  traffic light is not red, 'traffic_waypoint' is expected to be -1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        self.wp_traffic_light = WP_UNDEFINED

        # Subscribe to 'obstacle_waypoint' topic
        # (Index of waypoint closest to the next obstacle. If there is no
        #  obstacle ahead, 'obstacle_waypoint' is expected to be -1)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        self.wp_obstacle = WP_UNDEFINED

        # Subscribe to 'current_velocity' topic
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        self.current_velocity = 0.0

        # Publish waypoints ahead of the vehicle 
        # (Starting with the waypoint just ahead of the vehicle)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, 
                                                   queue_size = 1)
        self.final_waypoints = Lane()
        for i in range(LOOKAHEAD_WPS):
            self.final_waypoints.waypoints.append(Waypoint())

        # Reset parameters
        self.decel_max = 0.0
        self.accel_max = 0.0
        self.velocity_max = 0.0

        # Start node
        rospy.spin()

    def publish_waypoints(self):
        """ Publishes new waypoints for the waypoint follower node (starting
            with the next waypoint for the ego vehicle).
        """
        if (not len(self.waypoints) or not len(self.waypoint_velocities)):
            return

        # Find waypoint closest to ego vehicle
        # (adapted from Udacity SDC-ND Path Planning Project Starter Code, 
        #  accessed: 03/24/2018)
        # TODO: Reduce search interval based on (known) vehicle motion and
        #       previously selected path points (assuming the vehicle follows
        #       the trajectory)
        ego_pos = self.get_position(self.ego_pose)
        closest_id = 0
        closest_dist = sys.float_info.max
        for i, wp in enumerate(self.waypoints):
            # Calculate distance
            dist = self.distance(ego_pos, self.get_position(wp))  
            if (dist < closest_dist):
                closest_id = i
                closest_dist = dist

        if VERBOSE:
          rospy.loginfo('Closest waypoint (%i/%i) (dist: %.2f m): %s ',
                        closest_id, len(self.waypoints), closest_dist,
                        self.get_waypoint_string(self.waypoints[closest_id]))

        # Determine vehicle yaw (from quarternion representation)
        # (adapted from https://answers.ros.org/question/69754/quaternion-
        #    transformations-in-python/?answer=69799#post-id-69799,
        #  accessed 03/26/2018)
        ego_orient = self.get_orientation(self.ego_pose)
        q = (ego_orient.x, ego_orient.y, ego_orient.z, ego_orient.w)
        euler_angles = tf.transformations.euler_from_quaternion(q)
        ego_yaw = euler_angles[2]

        # Check if the closest waypoint is in front or behind of ego 
        # vehicle (consider only 2D (x, y) projection)
        # (adapted from Udacity Path Planning Project (Starter Code) and
        #  code posted by Jeremy Owen in the Udacity Self-Driving Car Slack on
        #  08/13/2017, 5:10 AM, https://carnd.slack.com/archives/C5ZS5SBA8/
        #    p1502593836232659, accessed: 03/10/2018).
        first_id = closest_id
        pos_closest = self.get_position(self.waypoints[closest_id])
        heading = math.atan2(pos_closest.y - ego_pos.y, 
                             pos_closest.x - ego_pos.x)
        heading = math.fmod(heading + 2 * math.pi, 2 * math.pi)
        ego_yaw = math.fmod(ego_yaw + 2 * math.pi, 2 * math.pi)
        angle = math.fabs(ego_yaw - heading)
        if (angle > math.pi):
            angle = 2 * math.pi - angle;
        if (angle > math.pi / 2):
            # Waypoint is behind vehicle --> select next
            first_id = (first_id + 1) % len(self.waypoints)

            if VERBOSE:
                first_wp = self.waypoints[first_id]
                first_dist = self.distance(ego_pos, 
                                           self.get_position(first_wp))
                rospy.loginfo('Next waypoint (%i/%i) (dist: %.2f m): %s',
                              first_id, len(self.waypoints), first_dist,
                              self.get_waypoint_string(first_wp))

        if VERBOSE:
            wp_selection = "(closest)"
            if (first_id != closest_id):
                wp_selection = "(next)"
            rospy.loginfo('First WP: heading(%.2f) <> yaw(%.2f) => %.2f %s', 
                          heading, ego_yaw, angle, wp_selection)

        # Create list of next waypoints (consider track wrap-around)
        # (update waypoint velocities in the process)
        self.final_waypoints.header.stamp = self.ego_pose.header.stamp
        last_velocity = self.current_velocity
        last_position = ego_pos
        for i in range(LOOKAHEAD_WPS):
            # Set waypoint
            wp_id = (first_id + i) % len(self.waypoints)
            self.final_waypoints.waypoints[i] = self.waypoints[wp_id]

            # Get distance between last position and waypoint
            wp_position = self.get_position(self.final_waypoints.waypoints[i])
            dist = self.distance(last_position, wp_position)

            # Determine next velocity
            # (adapted from waypoint_loader.py)
            wp_velocity = self.waypoint_velocities[wp_id]
            velocity_diff = wp_velocity - last_velocity
            if velocity_diff < 0.0:
                # Decelerate
                min_velocity = max(0.0, last_velocity 
                                        - math.sqrt(2 * self.decel_max * dist))
                wp_velocity = max(min_velocity, wp_velocity)
            elif velocity_diff > 0.0:
                # Accelerate
                max_velocity = min(self.velocity_max, 
                                   last_velocity + math.sqrt(2 * 
                                                       self.accel_max * dist))
                wp_velocity = min(max_velocity, wp_velocity)

            # Set waypoint velocity
            self.set_waypoint_velocity(self.final_waypoints.waypoints[i],
                                       wp_velocity)

            # Next (consider track wrap-around)
            last_velocity = wp_velocity
            last_position = wp_position

        # Publish next waypoints
        self.final_waypoints_pub.publish(self.final_waypoints)

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
        for wp in waypoints.waypoints:
            self.waypoint_velocities.append(self.get_waypoint_velocity(wp))
        self.decel_max = -rospy.get_param('/dbw_node/decel_limit')
        self.accel_max = rospy.get_param('/dbw_node/accel_limit')
        self.velocity_max = rospy.get_param('/waypoint_loader/velocity') / 3.6

    def traffic_cb(self, wp_traffic_light):
        """ Receives the index of the waypoint that corresponds to the
            stopline of the next red traffic light. An index of 'WP_UNDEFINED' 
            signals that no red traffic light (TL) is ahead (or close by)
            
            Arguments:
              wp_traffic_light -- Index of waypoint close to next TL stopline
        """
        self.wp_traffic_light = wp_traffic_light.data
        if self.wp_traffic_light != WP_UNDEFINED:
            self.check_waypoint_id(self.waypoints, self.wp_traffic_light)

            if VERBOSE:
                rospy.loginfo('Traffic light update (%i): %s', 
                              self.wp_traffic_light,
                              self.get_position_string(
                                  self.waypoint[self.wp_traffic_light]))

    def obstacle_cb(self, wp_obstacle):
        """ Receives the index of the waypoint that corresponds to the next 
            obstacle. An index of 'WP_UNDEFINED' signals that no obstacle is
            ahead (or close by)
            
            Arguments:
              wp_obstacle -- Index of waypoint close to next obstacle
        """
        self.wp_obstacle = wp_obstacle.data
        if self.wp_obstacle != WP_UNDEFINED:
            self.check_waypoint_id(self.waypoints, self.wp_obstacle)

            if VERBOSE:
                rospy.loginfo('Obstacle update (%i): %s', 
                              self.wp_obstacle,
                              self.get_position_string(
                                  self.waypoint[self.wp_obstacle]))

    def velocity_cb(self, twist):
        """ Receives the current ego vehicle twist from the simulator/vehicle 
            and extracts the current velocity

            Arguments:
              twist -- Ego vehicle twist
        """
        self.current_velocity = twist.twist.linear.x

    def check_waypoint_id(self, waypoints, wp_id):
        """ Check if waypoint ID is valid. Triggers an assert when not.

            Arguments:
              waypoints -- List of waypoints
              wp_id -- ID that should be checked
        """
        assert (wp_id >= 0 and wp_id < len(waypoints)),\
                "Invalid waypoint id (%i)" % wp_id

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

    def get_waypoint_velocity(self, wp):
        """ Get target velocity for waypoint

            Arguments:
              wp -- Target waypoint

            Return:
              Target velocity for waypoint
        """
        return wp.twist.twist.linear.x

    def set_waypoint_velocity(self, wp, velocity):
        """ Set the target velocity of a given waypoint (in place)
            
            Arguments:
              wp -- Waypoint
              velocity -- Target velocity
        """
        wp.twist.twist.linear.x = velocity

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
        """ Get the distance between two waypoints (by summing up the Euclidean 
            distances of the waypoints in between)

            Arguments:
              waypoints -- List of waypoints
              wp1 -- Index of the first waypoint
              wp2 -- Index of the second waypoint

            Return:
              The distance between the given waypoints
        """
        self.check_waypoint_id(waypoints, wp1)
        self.check_waypoint_id(waypoints, wp2)
        assert wp1 < wp2, ("Cannot get distance between waypoints"
                           " (invalid interval: %i - %i)") % (wp1, wp2)
        dist = 0
        for i in range(wp1, wp2):
            dist += self.distance(self.get_position(waypoints[i]), 
                                  self.get_position(waypoints[i + 1]))
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
