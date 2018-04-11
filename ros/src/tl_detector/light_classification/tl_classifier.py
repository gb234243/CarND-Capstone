# TODO: Uncomment this for production code
#from styx_msgs.msg import TrafficLight
#import rospkg

import cv2
import numpy as np
import tensorflow as tf

# TODO: Evaluate using Keras 
#from keras.models import load_model

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor

from utils import label_map_util

USE_COCO_MODEL = True
ENABLE_VISUAL = True

class TLClassifier(object):
    def __init__(self):
        # TODO: Uncomment for production code
        # Store the path
        #model_path = rospkg.RosPack().get_path('tl_detector')
        path = '../models'

        if USE_COCO_MODEL:
            # Use already trained model on COCO image set
            model_path = path + '/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
            label_path = path + '/ssd_mobilenet_v2_coco_2018_03_29/mscoco_label_map.pbtxt'
        else:
            model_path = path + '/pursuit/frozen_inference_graph.pb'
            label_path = path + '/pursuit/pursuit_label_map.pbtxt'
        pass

        # Load the graph model
        self.graph = self.load_graph(model_path)

        # Create TF sessions
        self.sess = tf.Session(graph=self.graph)

        # Definite input and output Tensors for graph
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')

        # This is the class from dataset
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')


        # Colors (one for each class)
        if ENABLE_VISUAL:
            cmap = ImageColor.colormap
            print("Number of colors =", len(cmap))
            self.COLOR_LIST = sorted([c for c in cmap.keys()])

        # Load the label file
        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
          
        # Actual detection.
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                            feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.8
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = image.size
        box_coords = self.to_image_coords(boxes, height, width)

        if ENABLE_VISUAL:
            # Each class with be represented by a differently colored box
            self.draw_boxes(image, box_coords, classes)
            image.save('./out.jpg')

        cls = classes.tolist()

        if USE_COCO_MODEL:
            # Class 10 is traffic light in COCO model dataset
            idx = next((x for x, i in enumerate(cls) if i == 10.), None)

            if idx == None:
                print("No traffic light found")
            else:
                for i in enumerate(cls):
                    print(self.category_index[i[1]]['name'])
        else:
            # class 2 is the red light in pursuit model
            idx = next((x for x, i in enumerate(cls) if i == 2.), None)

            if idx == None:
                print("No traffic light found")
            else:
                for i in enumerate(cls):
                    print(self.category_index[i[1]]['name'])

        # TODO Uncomment for the production code
        #return TrafficLight.Green

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

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        
        return box_coords

    def draw_boxes(self, image, boxes, classes, thickness=4):
        if ENABLE_VISUAL:
            """Draw bounding boxes on the image"""
            draw = ImageDraw.Draw(image)
            for i in range(len(boxes)):
                bot, left, top, right = boxes[i, ...]
                class_id = int(classes[i])
                color = self.COLOR_LIST[class_id]
                draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

if __name__ == '__main__':
    tl_cls = TLClassifier()

    # Load a sample image.
    image = Image.open('./TL.jpg')
    image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
    tl_cls.get_classification(image)

    