
############################################
## COPY DATASET
############################################
git checkout origin/Dataset

# Make a copy of the dataset
mkdir ../dataset_branch
cp -rf * ../dataset_branch

# Move the dataset to the Object_Detection branch
git checkout origin/Object_Detection
mv -f ../dataset_branch/* ./

# Remove the directory used to store copy of the dataset
rm -rf ../dataset_branch

############################################
## CLONE MODELS REPO
############################################
git clone https://github.com/tensorflow/models.git

# Perform installation steps
cd models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

############################################
## CREATE WORKSPACE
############################################
cd ../..

# Create workspace and copy files
# Make sure not to copy the cloned models repository
mkdir models/research/obj_detection_wkspace
mkdir models/research/obj_detection_wkspace/configs
cp * models/research/obj_detection_wkspace/
cp -rf configs models/research/obj_detection_wkspace/

# Change working directory
cd models/research/obj_detection_wkspace

############################################
## DOWNLOAD PRETRAIEND MODELS
############################################
mkdir models_pretrained
cd models_pretrained

# Download the Faster RCNN Pretrained Model
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_resnet50_coco_2018_01_28.tar.gz

# Download the SSD Inception v2 Pretrained Model
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
tar -xvzf ssd_inception_v2_coco_2017_11_17.tar.gz

############################################
## CREATE TF RECORD
############################################
python create_tfrecord_from_yaml.py