#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from scipy.spatial import KDTree
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 2
UPDATE_RATE = 5     # Update rate of the main loop (in Hz)

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.has_image = False
        self.camera_image = None
        self.lights = []

        '''
        /vehicle/traffic_lights provides you with the location of the traffic 
        light in 3D map space and helps you acquire an accurate ground truth 
        data source for the traffic light classifier by sending the current 
        color state of all traffic lights in the simulator. When testing on the 
        vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, 
                                self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', 
                                                      Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.is_init = False

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Adapted from Udacity SDC-ND Programming a Real Self-Driving Car 
        # Project Walkthrough (Term 3)
        self.waypoint_tree = None

        # Run main loop
        self.loop()

    def loop(self):
        """Main loop (check images at a fixed rate)

           (adapted from Udacity SDC-ND Programming a Real Self-Driving Car 
            Project Walkthrough (Term 3))
        """
        rate = rospy.Rate(UPDATE_RATE)
        while not rospy.is_shutdown():
            if (self.is_init):
                self.check_image()
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        if self.waypoints:
            return
        self.waypoints = waypoints

        # Adapted from Udacity SDC-ND Programming a Real Self-Driving Car 
        # Project Walkthrough (Term 3)
        waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y]\
                           for wp in waypoints.waypoints]
        self.waypoint_tree = KDTree(waypoints_2d)
        
        # Signal that node is initialized
        self.is_init = True

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Stores the current camera image

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

    def check_image(self):
        """Identifies red lights in camera images and publishes the 
           index of the waypoint closest to the red light's stop line to
           /traffic_waypoint
        """
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem

           (adapted from Udacity SDC-ND Programming a Real Self-Driving Car 
            Project Walkthrough (Term 3))
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.waypoint_tree.query([x, y], 1)[1]

    def get_light_state(self):
        """Determines the current color of the traffic light

           (adapted from Udacity SDC-ND Programming a Real Self-Driving Car 
            Project Walkthrough (Term 3))

        Returns:
            int: ID of traffic light color 
                 (specified in styx_msgs/TrafficLight)

        """

        # Check for valid image. If not present return RED as the state
        if(not self.has_image):
            return TrafficLight.RED

        # Convert the ROS image to CV2 for the classifer.
        img_cv = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        # Get the classifer
        return self.light_classifier.get_classification(img_cv)
        
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines 
           its location and color

           (adapted from Udacity Programming a Real Self-Driving Car Project 
            Walkthrough (Term 3))

        Returns:
            int: index of waypoint closest to the upcoming stop line for a 
                 traffic light (-1 if none exists)
            int: ID of traffic light color 
                 (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of 
        # for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x,
                                                   self.pose.pose.position.y)

            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state()
            return line_wp_idx, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
