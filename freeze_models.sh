cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Freeze Faster RCNN Model
python object_detection/export_inference_graph.py --pipeline_config_path=obj_detection_wkspace/configs/faster_rcnn_resnet50.config --trained_checkpoint=obj_detection_wkspace/models_trained/faster_rcnn_sim/model.ckpt-200000 --output_directory=obj_detection_wkspace/models_frozen/faster_rcnn_sim

## Freeze SSD Inception V2 Model
#python object_detection/export_inference_graph.py --pipeline_config_path=obj_detection_wkspace/configs/ssd_inception_v2.config --trained_checkpoint=obj_detection_wkspace/models_trained/ssd_inception_sim/model.ckpt-200000 --output_directory=obj_detection_wkspace/models_frozen/ssd_inception_sim
