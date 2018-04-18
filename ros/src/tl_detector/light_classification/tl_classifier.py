from styx_msgs.msg import TrafficLight
import rospkg

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image

class TLClassifier(object):
    def __init__(self):
        # Store the path
        model_path = rospkg.RosPack().get_path('light_classification')

        # Use already trained model on COCO image set
        model_path += '/models/frozen_pursuit_model.pb'

        # Load the graph model
        self.graph = self.load_graph(model_path)

        # Create TF sessions
        self.sess = tf.Session(graph=self.graph)

        # Definite input and output Tensors for graph
        self.image_tensor = self.graph.get_tensor_by_name('input_images:0')
        self.output = self.graph.get_tensor_by_name('output:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        (output) = self.sess.run([self.output], feed_dict={self.image_tensor: image_np})
        tl_color = np.squeeze(output)

        #['NoLight', 'Red', 'Yellow', 'Green']
        if tl_color[1] == 1.:
            print("RED")
            return TrafficLight.RED
        elif tl_color[0] == 1.:
            print("Unknown")
            return TrafficLight.UNKNOWN
        elif tl_color[2] == 1.:
            print("Yellow")
            return TrafficLight.YELLOW
        elif tl_color[3] == 1.:
            print("GREEN")
            return TrafficLight.GREEN
        else:
            print("Unknown")
            return TrafficLight.UNKNOWN

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

if __name__ == '__main__':
    tl_cls = TLClassifier()

    # Load a sample image.
    image = Image.open('./green.jpg')
    tl_cls.get_classification(image)

    