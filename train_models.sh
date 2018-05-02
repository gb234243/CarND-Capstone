# Make folders if necessary
mkdir models_trained
mkdir models_trained/faster_rcnn_sim
mkdir models_trained/ssd_inception_sim

# Change parent working directory
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Train Faster RCNN Model
python object_detection/train.py --pipeline_config_path=obj_detection_wkspace/configs/faster_rcnn_resnet50.config --train_dir=obj_detection_wkspace/models_trained/faster_rcnn_sim

## Train SSD Inception V2 Model
#python object_detection/train.py --pipeline_config_path=obj_detection_wkspace/configs/ssd_inception_v2.config --train_dir=obj_detection_wkspace/models_trained/ssd_inception_sim
