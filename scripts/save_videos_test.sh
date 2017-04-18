#!/bin/bash

if [ -z "$1" ]
  then
    echo "No folder supplied!"
    echo "Usage: bash `basename "$0"` vot_videos_folder"
    exit
fi

set -x


VIDEOS_FOLDER=$1
# GPU_ID is 2nd cmd line parameter, defaults to 0
GPU_ID=${2:-0}

# Used to specify source of weights, 3nd cmd line parameter, defaults to 0
GPU_ID_WEIGHTS=${3:-0}


DEPLOY_PROTO=nets/tracker.prototxt

# DO NOT COMMIT
# Original
# CAFFE_MODEL=nets/models/pretrained_model/tracker.caffemodel
# Rotation:
# DO NOT COMMIT
# TODO: take the weights with the latest timestamp
CAFFE_MODEL=nets/solverstate/GOTURN${GPU_ID_WEIGHTS}/caffenet_train_iter_20000.caffemodel
# CAFFE_MODEL=nets/solverstate/GOTURN0/caffenet_train_iter_80000.caffemodel

OUTPUT_FOLDER=nets/tracker_output/GOTURN${GPU_ID_WEIGHTS}_test

echo "Saving videos to " $OUTPUT_FOLDER

# Run tracker on test set and save videos
build/save_videos_vot $VIDEOS_FOLDER $DEPLOY_PROTO $CAFFE_MODEL $OUTPUT_FOLDER $GPU_ID |& tee  $OUTPUT_FOLDER/results.txt
# |& redirect stdout and stderr to stdout
# tee writs output to file and console

# scripts/save_videos_test.sh tmp/vot2014

# build/save_videos_vot tmp/vot2014 nets/tracker.prototxt nets/solverstate/GOTURN0/caffenet_train_iter_50000.caffemodel nets/tracker_output/GOTURN0_test 0
