# Udacity Capstone - Object Detection Model Training

## Requirements
* TensorFlow v1.5.0 or later (for training)
* TensorFlow v1.3.0 (for freezing the model)

## Instructions

### Setup
Clone this repo and make it the parent working directory.

```bash
$ git clone https://github.com/gb234243/CarND-Capstone.git
$ cd CarND-Capstone
```

### Move the Dataset to the Object_Detection branch
To train the model, the dataset needs to be converted to a TF record file. The TF record file requires utilities from the TensorFlow models repo which we will clone later. The dataset is already in the Dataset branch, so let's copy it over to our Object_Detection branch

```bash
$ git checkout origin/Dataset
$ 
$ # Make a copy of the dataset
$ mkdir ../dataset_branch
$ cp -rf * ../dataset_branch
$ 
$ # Move the dataset to the Object_Detection branch
$ git checkout origin/Object_Detection
$ mv -f ../dataset_branch/* ./
$ 
$ # Remove the directory used to store copy of the dataset
$ rm -rf ../dataset_branch
```

### Clone the TensorFlow Models Repo
The TensorFlow models repo provides pretrained models which we can use to create an custom object detection model. Once cloned, make `models/research` the parent working irectory and follow the [install instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

```bash
$ git clone https://github.com/tensorflow/models.git
$ 
$ # Perform installation steps
$ cd models/research
$ 
```

### Creating a Workspace
To train the models, `models/research` needs to be the parent working directory. To ensure pathnames are short and easy to figure out, a workspace directory will be created with the files in our repo's root directory. If you choose to use a name other than `obj_detection_wkspace`, please make sure to modify the configuration files and shell files in the next few steps.

```bash
$ cd ../..
$ 
$ # Create workspace and copy files
$ # Make sure not to copy the cloned models repository
$ mkdir models/research/obj_detection_wkspace
$ mkdir models/research/obj_detection_wkspace/configs
$ cp * models/research/obj_detection_wkspace/
$ cp -rf configs models/research/obj_detection_wkspace/
$ mv -f data models/research/obj_detection_wkspace/
$ 
$ # Change working directory
$ cd models/research/obj_detection_wkspace
```

### Create TF Record from Dataset
The TF record can now be created from within the workspace. A notebook, `Create TF Record from YAML.ipynb`, with a corresponding python file, `create_tfrecord_from_yaml.py`, has been provided to do that. Run either to completion to generate the `data.record` file. If you plan on using your own data, make sure to edit this notebook.

Note that data should be added in a random order into the TF record file. If the TF record data is not shuffled (i.e. classes are grouped such that only class_x appears at the end of the TF record), the final trained network may end up unlearning all other classes and focus on learning just the last class it sees.

### Download Pretrained Models
Pretrained models need to be downloaded for the training. These pretrained models provide weights whcih ensure that training will take less time than without the weights. Other models can be downloaded from the [object detection models zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

```bash
$ mkdir models_pretrained
$ cd models_pretrained
$ 
$ # Download the Faster RCNN Pretrained Model
$ wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
$ tar -xvzf faster_rcnn_resnet50_coco_2018_01_28.tar.gz
$ 
$ # Download the SSD Inception v2 Pretrained Model
$ wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
$ tar -xvzf ssd_inception_v2_coco_2017_11_17.tar.gz
```

### Edit Configuration Files
Configuration files are already provided in the `configs` folder. Downloaded pretrained models also have configuration files as well. If you plan on using your own configurations, please make sure, at the very least, to edit the following parameters.

* model.\*.num_classes
* model.\*.image_resizer
* model.\*.second_stage_post_processing.max_detections_per_class
* model.\*.second_stage_post_processing.max_total_detections
* model.\*.second_stage_batch_size
* train_config.fine_tune_checkpoint
* train_config.batch_size
* train_config.num_steps
* train_input_reader.tf_record_input_reader
* train_input_reader.label_map_path
* eval_input_reader.tf_record_input_reader
* eval_input_reader.label_map_path

Note that `second_stage_batch_size` needs to be less than `second_stage_post_processing.max_total_detections`. Otherwise, the `object_detection/train.py` file will complain.

`batch_size` will need to be changed so that it can run on your system. `num_steps` may also need to be changed to reflect the new `batch_size` value.

### Create a Label Map File
Training also requires a label map. A label map is provided at `configs/label_map.pbtxt`. For your own labels, create a new `label_map.pbtxt` file that defines each label and a corresponding id number as shown below.

```
item {
	id: ###
	name: 'ClassName'
}
```

### Training a Model
Now that everything is ready, it is time to train the model. `train_models.sh` is provided to train either a Faster RCNN or SSD model. Comment out the lines as desired.

```bash
$ cd ..
$
$ # Run training
$ ./train_models.sh
```

However, if you wish to run the commands line-by-line yourself, run the following commands. Note that the `train.py` file expects the parent working directory to be the `models/research` directory.

```bash
$ cd ..
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$
$ # Train model
$ python object_detection/train.py --pipeline_config_path=path/to/model.config --train_dir=path/to/dir/to/save/model
``` 

On a machine with an Nvidia GTX 1080, training with the Faster RCNN configuration takes 9 hours. The SSD configuration takes 13 hours. During training, there will be a printout of the time it takes to run one training step. You can approximate the time it will take by multiplying this value with your `batch_size` parameter.

Note that you can stop the training midway if necessary. Calling `train.py` with the defined parameters and train directory will cause the training to resume from its last saved checkpoint.

### Freezing a Model
Once the model has been trained, it's time to freeze it. The provided `freeze_models.sh` will freeze either a Faster RCNN or SSD model. Comment out the lines as desired.

```bash
$ # Freeze models
$ freeze_models.sh
```

If you wish to run the commands line-by-line yourself, run the following commands. Note that like `train.py`, the `export_inference_graph.py` file expects the parent working directory to be the `models/research` directory.

```bash
$ # Freeze model
$ python object_detection/export_inference_graph.py --pipeline_config_path=path/to/model.config --trained_checkpoint=path/to/trained/model.ckpt-#### --output_directory=path/to/dir/to/save/frozen/model
``` 