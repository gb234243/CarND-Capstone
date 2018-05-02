# Udacity Capstone Dataset

The Dataset branch contains the relevant dataset, XML, and YAML files necessary for training an object detection model from TensorFlow's object detection zoo.

Note that there is a discrepancy between the file format and file paths listed in the XML file and the provided image file format (currently JPG). Originally, the image files were PNG files. They have been changed to JPG files to save disk space. The `Create YAML from XML files.ipynb` provides code to convert the PNG files to JPG and remove all PNG files. The `data` directory used to be called `final` and is still listed as `final` in the XML files.

## Obtaining Dataset Images
The images in this dataset are obtained from the capstone simulator. Using the provided traffic light colors, the images are grouped into either a red, green, or yellow directory. This is done to make it easier to separate the task of defining bounding boxes.

## Defining Bounding Boxes
The [LabelImg](https://github.com/tzutalin/labelImg) program is used to create bounding boxes for all objects in the images. Each image's annotated bounding boxes are saved as XML files.

## Creating the YAML file
A YAML file can be created to combine all the XML annotated data into a single file. The `Create YAML from XML files.ipynb` notebook provides code that goes through each XML file and creates the desired YAML file. The notebook can be updated to output YAML files that follow a custom schema.