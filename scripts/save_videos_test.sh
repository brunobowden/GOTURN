#!/bin/bash

if [ -z "$1" ]
  then
    echo "No folder supplied!"
    echo "Usage: bash `basename "$0"` vot_videos_folder"
    exit
fi


VIDEOS_FOLDER=${1}
# GPU_ID is, 2nd parameter, default 0
GPU_ID=${2:-0}
# Source of weights, 3rd parameter, defaults 0
GPU_ID_WEIGHTS=${3:-0}

DEPLOY_PROTO=nets/tracker.prototxt

# DO NOT COMMIT
# Original
# CAFFE_MODEL=nets/models/pretrained_model/tracker.caffemodel
# Latest model for a given GPU
CAFFE_MODEL=`find nets/solverstate/GOTURN${GPU_ID_WEIGHTS}/caffenet_train_iter_*.caffemodel -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "`

OUTPUT_FOLDER=nets/tracker_output/GOTURN${GPU_ID_WEIGHTS}_test

echo "Videos Output: " $OUTPUT_FOLDER

# Run tracker on test set and save videos
CMD="build/save_videos_vot $VIDEOS_FOLDER $DEPLOY_PROTO $CAFFE_MODEL $OUTPUT_FOLDER $GPU_ID |& tee $OUTPUT_FOLDER/results.txt"
echo CMD: $CMD
eval $CMD
# |& redirect stdout and stderr to stdout
# tee writs output to file and console

# bash scripts/save_videos_test.sh tmp/vot2014

# build/save_videos_vot tmp/vot2014 nets/tracker.prototxt nets/solverstate/GOTURN0/caffenet_train_iter_50000.caffemodel nets/tracker_output/GOTURN0_test 0
