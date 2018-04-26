from styx_msgs.msg import TrafficLight
import rospkg

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image

class TLClassifier(object):
    def __init__(self):
        # Store the path
        model_path = rospkg.RosPack().get_path('tl_detector')

        # Use already trained model on COCO image set
        model_path += '/light_classification/model/tl-classifier-frozen-opt.pb'

        # Load the graph model
        self.graph = self.load_graph(model_path)

        # Create TF sessions
        self.sess = tf.Session(graph=self.graph)

        # Define input and output Tensors for graph
        self.image_tensor = self.graph.get_tensor_by_name('input_images:0')
        self.output = self.graph.get_tensor_by_name('output:0')
        #for p in self.graph.get_operations():
        #    print(p.name)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color 
                 (specified in styx_msgs/TrafficLight)
        """
        # Resize and normalize the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(cv2.resize(image, (300, 300)))
        image = image / 128.0 - 1.0
        idx = self.sess.run(tf.argmax(self.output, 1), 
                            feed_dict={self.image_tensor: [image]})[0]

        #['NoLight', 'Red', 'Yellow', 'Green']
        if idx == 1:
            print("Red")
            return TrafficLight.RED
        elif idx == 0:
            print("Unknown")
            return TrafficLight.UNKNOWN
        elif idx == 2:
            print("Yellow")
            return TrafficLight.YELLOW
        elif idx == 3:
            print("Green")
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
    image = Image.open('./Images/simulator/final/0045.jpg')
    tl_cls.get_classification(image)

    
