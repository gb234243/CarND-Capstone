from styx_msgs.msg import TrafficLight
import rospkg

import cv2
import numpy as np
import tensorflow as tf
import time

from PIL import Image

class TLClassifier(object):
    def __init__(self):
        # Store the path
        model_path = rospkg.RosPack().get_path('tl_detector')
        model_path += '/light_classification/model/frozen_faster_rcnn_sim.pb'
        # model_path += '/light_classification/model/frozen_ssd_inceptionv2_sim.pb'

        starttime = time.time()
        self.graph = self.load_graph(model_path)

        with self.graph.as_default():
            sess = tf.Session()
            
            # Get the graph and all its tensor names
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}

            # Get the tensors of interest
            tensor_dict = {}
            for key in ['num_detections', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            # Get input tensor
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Store session and 
            self.sess = sess
            self.image_tensor = image_tensor
            self.tensor_dict = tensor_dict
        endtime = time.time()
        print("Classifier intitializiation took {:.5f} seconds".format(endtime - starttime))

        # Attempt to lose the 2 second start time...
        # self.get_classification(np.zeros((600, 800, 3), dtype=np.uint8)) 

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color 
                 (specified in styx_msgs/TrafficLight)
        """
        # Convert the Image's Color Space
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict Traffic Lights inside the Image
        starttime = time.time()
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: [image]})
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        # Process all Predictions and get single prediction
        # Get all valid predictions
        # If there's only one valid prediction, then that's the image's prediction value
        idx = 0
        guesses = set()
        for index in range(output_dict['num_detections']):
            if output_dict['detection_scores'][index] > 0.7:
                guesses.add(output_dict['detection_classes'][index])

        if len(guesses) == 1:
            idx = list(guesses)[0]
        
        endtime = time.time()
        print("Inferencing took {:.5f} secs".format(endtime - starttime))

        if idx == 2:
            print("Red")
            return TrafficLight.RED
        elif idx == 3:
            print("Yellow")
            return TrafficLight.YELLOW
        elif idx == 4:
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

