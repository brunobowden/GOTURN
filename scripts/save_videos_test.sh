#!/bin/bash

if [ -z "$1" ]
  then
    echo "No folder supplied!"
    echo "Usage: bash `basename "$0"` vot_videos_folder"
    exit
fi

set -ex

# Choose which GPU the tracker runs on
GPU_ID=0

VIDEOS_FOLDER=$1

FOLDER=GOTURN0_test

DEPLOY_PROTO=nets/tracker.prototxt

# DO NOT COMMIT
# Original
# CAFFE_MODEL=nets/models/pretrained_model/tracker.caffemodel
# Rotation:
CAFFE_MODEL=nets/solverstate/GOTURN0/caffenet_train_iter_10000.caffemodel
# CAFFE_MODEL=nets/solverstate/GOTURN0/caffenet_train_iter_21000.caffemodel

OUTPUT_FOLDER=nets/tracker_output/$FOLDER

echo "Saving videos to " $OUTPUT_FOLDER

# Run tracker on test set and save videos
build/save_videos_vot $VIDEOS_FOLDER $DEPLOY_PROTO $CAFFE_MODEL $OUTPUT_FOLDER $GPU_ID |& tee  $OUTPUT_FOLDER/results.txt
