#!/bin/bash

if [ -z "$1" ]
  then
    echo "No folder supplied!"
    echo "Usage: bash `basename "$0"` vot_videos_folder"
    exit
fi

set -x

# Choose which GPU the tracker runs on
GPU_ID=0

VIDEOS_FOLDER=$1

FOLDER=GOTURN1_test

DEPLOY_PROTO=nets/tracker.prototxt

# DO NOT COMMIT
# Original
# CAFFE_MODEL=nets/models/pretrained_model/tracker.caffemodel
# Rotation:
# CAFFE_MODEL=nets/solverstate/GOTURN1/caffenet_train_iter_1000.caffemodel
CAFFE_MODEL=nets/solverstate/GOTURN1/caffenet_train_iter_21000.caffemodel

OUTPUT_FOLDER=nets/tracker_output/$FOLDER

echo "Saving videos to " $OUTPUT_FOLDER

# Run tracker on test set and save vidoes 
build/save_videos_vot $VIDEOS_FOLDER $DEPLOY_PROTO $CAFFE_MODEL $OUTPUT_FOLDER $GPU_ID 
