#!/usr/bin/env python

from collections import deque
from geometry_msgs.msg import Pose, PoseStamped, TwistStamped
import math
import rospy
from scipy.spatial import KDTree
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
import sys
import tf

'''
This node will publish waypoints from the car's current position to some `x`
distance ahead.
'''

# Constants

# Number of waypoints we will publish (reduced as suggested by "alangordon" in
# the Udacity Self-Driving Car Slack on 04/05/2018, 8:28 AM,
# https://carnd.slack.com/archives/C6NVDVAQ3/p1522909695000091, accessed
# 04/05/2018)
LOOKAHEAD_WPS = 50

# Undefined waypoint index
WP_UNDEFINED = -1

# Update rate of the main loop (in Hz)
UPDATE_RATE = 30

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

        # Subscribe to 'base_waypoints' topic
        # (List of track waypoints; will only be send once)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.waypoints = None
        self.waypoint_velocities = []
        self.num_waypoints = 0
        self.is_init = False

        # From Udacity SDC-ND Programming a Real Self-Driving Car 
        # Project Walkthrough (Term 3)
        self.waypoint_tree = None

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
        self.loop()

    def loop(self):
        """Main loop (publish waypoints at a fixed rate)

           (adapted from Udacity SDC-ND Programming a Real Self-Driving Car 
            Project Walkthrough (Term 3))
        """
        rate = rospy.Rate(UPDATE_RATE)
        while not rospy.is_shutdown():
            if (self.is_init):
                self.publish_waypoints()

            rate.sleep()

    def publish_waypoints(self):
        """ Publishes new waypoints for the waypoint follower node (starting
            with the next waypoint for the ego vehicle).
        """
        # Get start position
        ego_pos = self.ego_pose.pose.position

        # Find waypoint closest to ego vehicle
        # (adapted from Udacity SDC-ND Programming a Real Self-Driving Car 
        #  Project Walkthrough (Term 3))
        closest_id = self.waypoint_tree.query([ego_pos.x, ego_pos.y], 1)[1]

        # Determine vehicle yaw (from quarternion representation)
        # (adapted from https://answers.ros.org/question/69754/quaternion-
        #    transformations-in-python/?answer=69799#post-id-69799,
        #  accessed 03/26/2018)
        ego_orient = self.ego_pose.pose.orientation
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
        pos_closest = self.waypoints[closest_id].pose.pose.position
        heading = math.atan2(pos_closest.y - ego_pos.y, 
                             pos_closest.x - ego_pos.x)
        heading = math.fmod(heading + 2 * math.pi, 2 * math.pi)
        ego_yaw = math.fmod(ego_yaw + 2 * math.pi, 2 * math.pi)
        angle = math.fabs(ego_yaw - heading)
        if (angle > math.pi):
            angle = 2 * math.pi - angle;
        if (angle > math.pi / 2):
            # Waypoint is behind vehicle --> select next
            first_id = (first_id + 1) % self.num_waypoints

        # Create list of next waypoints (consider track wrap-around)
        # (update waypoint velocities in the process)
        self.final_waypoints.header.stamp = self.ego_pose.header.stamp
        self.calculate_velocities(self.final_waypoints.waypoints, first_id,
                                  self.current_velocity, ego_pos)

        # Publish next waypoints
        self.final_waypoints_pub.publish(self.final_waypoints)

    def calculate_velocities(self, waypoints, first_id, last_velocity,
                             last_position):
        """ Set velocities for next waypoints for ego vehicle

            Arguments:
              waypoints -- Next waypoints for the ego vehicle (expected size:
                           LOOKAHEAD_WPS)
              first_id -- ID (absolute) of next waypoint for ego vehicle
              last_velocity -- Current velocity of ego vehicle
              last_position -- Current position of ego vehicle
        """
        stop_id = self.get_stop_point(waypoints, first_id)

        for i in xrange(LOOKAHEAD_WPS):
            # Set waypoint
            wp_id = (first_id + i) % self.num_waypoints
            waypoints[i] = self.waypoints[wp_id]

            # After stop point?
            if stop_id != WP_UNDEFINED and i >= stop_id:
                waypoints[i].twist.twist.linear.x = 0.0
                continue

            # Get distance between last position and waypoint
            wp_position = waypoints[i].pose.pose.position
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

            # Consider stop point
            if stop_id != WP_UNDEFINED:
                dist = self.distance_path(waypoints, i, stop_id)
                v_decel = math.sqrt(self.decel_max / 2 * dist)
                if v_decel < 1.0:
                    v_decel = 0.0
                wp_velocity = min(wp_velocity, v_decel)

            # Set waypoint velocity
            waypoints[i].twist.twist.linear.x = wp_velocity

            # Next (consider track wrap-around)
            last_velocity = wp_velocity
            last_position = wp_position

    def get_stop_point(self, waypoints, first_id):
        """ Check if next traffic light/obstacle is in range of next set of
            waypoints for ego vehicle

            Arguments:
              waypoints -- Next waypoints for the ego vehicle (expected size:
                           LOOKAHEAD_WPS)
              first_id -- ID (absolute) of next waypoint for ego vehicle

            Return:
              ID of stopping point in set of waypoints
        """
        # Make IDs relative
        obstacle_id = self.waypoint_in_range(self.wp_obstacle, first_id)
        traffic_light_id = self.waypoint_in_range(self.wp_traffic_light, 
                                                  first_id)

        # Stop?
        stop_id = obstacle_id
        if (traffic_light_id != WP_UNDEFINED and
            (stop_id == WP_UNDEFINED or traffic_light_id < stop_id)):
            stop_id = traffic_light_id
        return stop_id

    def waypoint_in_range(self, wp_id, first_id):
        """ Check if a given waypoint (defined through its absolute ID) is in 
            range of the list of next waypoints (where the first waypoint is
            defined through 'first_id')

            Arguments:
              wp_id -- Waypoint ID (absolute)
              first_id -- ID of first waypoint in 'waypoints' (absolute)

            Return:
              Relative position of the waypoint in the list of waypoints (when 
              the waypoint is in range). WP_UNDEFINED, otherwise.
        """
        if wp_id == WP_UNDEFINED:
            return WP_UNDEFINED

        # Make ID relative
        if wp_id < first_id:
            wp_id = self.num_waypoints + wp_id - first_id
        else:
            wp_id = wp_id - first_id

        # Request a full stop a few waypoints before the stop line
        # (to prevent driving over the stop line (e.g. due to latency from the
        #  controllers, node update rates, etc.) at which point the traffic
        #  light will not be detected in front of the car anymore)
        #wp_id = (wp_id - 4) if wp_id > 3 else 0 

        # Is the waypoint in range?
        if wp_id >= LOOKAHEAD_WPS:
            return WP_UNDEFINED

        return wp_id

    def pose_cb(self, ego_pose):
        """ Callback function for ego vehicle pose (position, orientation)  
            updates.

            Arguments:
              ego_pose: Current ego pose
        """
        self.ego_pose = ego_pose

    def waypoints_cb(self, waypoints):
        """ Receives a list of track waypoints and stores them internally
            
            Arguments:
              waypoints -- List of track waypoints
        """
        if self.waypoints:
            return

        self.waypoints = waypoints.waypoints
        self.num_waypoints = len(self.waypoints)
        for wp in waypoints.waypoints:
            self.waypoint_velocities.append(wp.twist.twist.linear.x)

        # Adapted from Udacity SDC-ND Programming a Real Self-Driving Car 
        # Project Walkthrough (Term 3)
        waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y]\
                           for wp in self.waypoints]
        self.waypoint_tree = KDTree(waypoints_2d)

        # Get limits
        self.decel_max = -rospy.get_param('/dbw_node/decel_limit')
        self.accel_max = rospy.get_param('/dbw_node/accel_limit')
        self.velocity_max = rospy.get_param('/waypoint_loader/velocity') / 3.6

        # Mark node as ready
        self.is_init = True

    def traffic_cb(self, wp_traffic_light):
        """ Receives the index of the waypoint that corresponds to the
            stopline of the next red traffic light. An index of 'WP_UNDEFINED' 
            signals that no red traffic light (TL) is ahead (or close by)
            
            Arguments:
              wp_traffic_light -- Index of waypoint close to next TL stopline
        """
        self.wp_traffic_light = wp_traffic_light.data
        #if self.wp_traffic_light != WP_UNDEFINED:
        #    self.check_waypoint_id(self.waypoints, self.wp_traffic_light)

    def obstacle_cb(self, wp_obstacle):
        """ Receives the index of the waypoint that corresponds to the next 
            obstacle. An index of 'WP_UNDEFINED' signals that no obstacle is
            ahead (or close by)
            
            Arguments:
              wp_obstacle -- Index of waypoint close to next obstacle
        """
        self.wp_obstacle = wp_obstacle.data
        #if self.wp_obstacle != WP_UNDEFINED:
        #    self.check_waypoint_id(self.waypoints, self.wp_obstacle)

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
        #self.check_waypoint_id(waypoints, wp1)
        #self.check_waypoint_id(waypoints, wp2)
        #assert wp1 < wp2, ("Cannot get distance between waypoints"
        #                   " (invalid interval: %i - %i)") % (wp1, wp2)
        dist = 0
        for i in range(wp1, wp2):
            dist += self.distance(waypoints[i].pose.pose.position, 
                                  waypoints[i + 1].pose.pose.position)
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
