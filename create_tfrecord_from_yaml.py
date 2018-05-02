import os, io, hashlib, logging

import tensorflow as tf
import PIL.Image
import yaml
import matplotlib.pyplot as plt

from object_detection.utils import dataset_util, label_map_util

############################################################
# Define YAML Keys
############################################################
def getCoords(box):
    xmin = box['xmin']
    xmax = xmin + box['x_width']
    ymin = box['ymin']
    ymax = ymin + box['y_height']
    
    return (xmin, ymin, xmax, ymax)

yaml_keys = {
    'path': 'filename',
    'boxes': 'annotations',
    'label': 'class',
    'getCoords': getCoords,
    'labelEnum': {
        'default': 1,
        'Red': 2,
        'Yellow': 3,
        'Green': 4
    }
}

############################################################
# Import YAML file
############################################################
yaml_filename = "data.yml"
dataset = yaml.load(open(yaml_filename, 'rb').read())

for i in range(len(dataset)):
    dataset[i][yaml_keys['path']] = os.path.abspath(os.path.join(
        os.path.dirname(yaml_filename), dataset[i][yaml_keys['path']]))

############################################################
# Create TF Record from YAML data record
############################################################
def tf_example(data):
    # Read in image and get encoded key value
    img_path = data[yaml_keys['path']]
    filename = os.path.basename(img_path)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    # Get image dimensions
    width = int(image.width)
    height = int(image.height)
    
    # Create lists of all bounding boxes
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    
    # Get lists of all bounding boxes
    for box in data[yaml_keys['boxes']]:
        # Get bounding box coordinates
        a, b, c, d = yaml_keys['getCoords'](box)
        xmin.append(float(a) / width)
        ymin.append(float(b) / height)
        xmax.append(float(c) / width)
        ymax.append(float(d) / height)
        
        # Get bounding box label
        label = box[yaml_keys['label']]
        if label in yaml_keys['labelEnum']:
            labelEnum = yaml_keys['labelEnum'][label]
        else:
            labelEnum = yaml_keys['labelEnum']['default']
        
        classes.append(labelEnum)
        classes_text.append(label.encode('utf8'))
        
    # Create TF example
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))
    
    # Return TF example
    return example

############################################################
# Save TF Record
############################################################
# Determine name of record based on YAML filename
output_name = '.'.join(
    os.path.basename(yaml_filename)
    .split('.')[:-1]) + '.record'
output_path = os.path.abspath(os.path.join(
    os.path.dirname(yaml_filename), output_name))

# Create and write the record to file
writer = tf.python_io.TFRecordWriter(output_path)
for data in dataset:
    example = tf_example(data)
    writer.write(example.SerializeToString())
writer.close()

