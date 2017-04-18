#!/bin/bash

if [ -z "$4" ]
  then
    echo "No folder supplied!"
    echo "Usage: bash `basename "$0"` imagenet_folder imagenet_annotations_folder alov_videos_folder alov_annotations_folder [gpu_id]"
    echo "GPU_ID defaults to 0. If specified, creates dedicated directory per GPU"
    exit
fi

set -x

VIDEOS_FOLDER_IMAGENET=$1
ANNOTATIONS_FOLDER_IMAGENET=$2
VIDEOS_FOLDER=$3
ANNOTATIONS_FOLDER=$4
# GPU_ID is 5th cmd line parameter, defaults to 0
GPU_ID=${5:-0}

RANDOM_SEED=800
FOLDER=GOTURN$GPU_ID
echo FOLDER: $FOLDER

SOLVER=nets/solver.prototxt
TRAIN_PROTO=nets/tracker.prototxt
# DO NOT COMMIT
# CAFFE_MODEL=nets/models/weights_init/tracker_init.caffemodel
# Start with pretrained weights and just focus on rotation training
CAFFE_MODEL=nets/models/pretrained_model/tracker.caffemodel

BASEDIR=nets
RESULT_DIR=$BASEDIR/results/$FOLDER
SOLVERSTATE_DIR=$BASEDIR/solverstate/$FOLDER

#Make folders to store results and snapshots
mkdir -p $RESULT_DIR
mkdir -p $SOLVERSTATE_DIR

#Modify solver to save snapshot in SOLVERSTATE_DIR
mkdir -p nets/solver_temp
SOLVER_TEMP=nets/solver_temp/solver_temp_$FOLDER.prototxt
sed s#SOLVERSTATE_DIR#$SOLVERSTATE_DIR# <$SOLVER >$SOLVER_TEMP
sed -i s#TRAIN_FILE#$TRAIN_PROTO# $SOLVER_TEMP
sed -i s#DEVICE_ID#$GPU_ID# $SOLVER_TEMP
sed -i s#RANDOM_SEED#$RANDOM_SEED# $SOLVER_TEMP

# Example generation based on modeling expected frame to frame changes
# See Section G: http://davheld.github.io/GOTURN/supplement.pdf
LAMBDA_SHIFT=5  # +/-(1/5) (translation of 20% of width, frame to frame)
LAMBDA_SCALE=15  # +/-(1/15) (width changes 7%, frame to frame)
MIN_SCALE=-0.4
MAX_SCALE=0.4
# According to Section G above, should use training set for distribution
# model and validation set for tuning parameters. In this case, the
# rotation is assumed to have a Laplace Distribution and the parameters
# are tuned manually from the ALOV videos with rotation. From manual
# observation, several videos do a full rotation approximately
# every second, which equates to 15 degrees frame to frame rotation.
# Parameter is chosen as fraction of kRotationRange in bounding_box.cpp
# param = (180 degrees range / 15 degrees) = 12
LAMBDA_ROTATION=12  # +/-(1/12) (15 degree rotation, frame to frame)
# Rotation Distribution
# Further concern is that while some videos show +/- 15 degree rotation
# most videso exhibit only small rotations. This is manually calculated.
#
# video id - approx. frames with rotation - name - notes
# 0 - 240 - ball
# 6 - 72 - diving - approx. full rotation per second
# 11 - 72 - gymnastics - approx. full rotation per second
# 15 - 72 - motocross - approx. full rotation per second
# 18 - 48 - hamster ball
# For videos 0 and 18, the rotation isn't parallel to screen
# and hard to even manually track.
#
# Percentage of frames with larger ~15 degree frame to frame rotation
#   = frames with large rotation / total frames
#   = (240 + 72 + 72 + 72 + 48) / 10189
#   = 504 / 10189
#   = 5% of the time
#
# The rotation distribution should be +/-15 degrees for 5% the time
# then 95% of the time much smaller. This 5 : 95 split isn't modeled
# here but is handled in bounding_box.cpp on line 374 with a 50 : 50
# split, to increase training speed during development
# if (sample_rand_uniform() < 0.5) {
#   rotation /= 10.0;
# }

echo LAMBDA_SCALE: $LAMBDA_SCALE
echo LAMBDA_SHIFT: $LAMBDA_SHIFT
echo LAMBDA_ROTATION: $LAMBDA_ROTATION


build/train $VIDEOS_FOLDER_IMAGENET $ANNOTATIONS_FOLDER_IMAGENET $VIDEOS_FOLDER $ANNOTATIONS_FOLDER $CAFFE_MODEL $TRAIN_PROTO $SOLVER_TEMP $LAMBDA_SHIFT $LAMBDA_SCALE $MIN_SCALE $MAX_SCALE $LAMBDA_ROTATION $GPU_ID $RANDOM_SEED |& tee $RESULT_DIR/results.txt
# |&  - pipe both stdout and stderr to stdout
# tee - output to console and write to results.txt

# DO NOT COMMIT
# bash scripts/train.sh tmp/ILSVRC2014/ILSVRC2014_DET_train_extracted/ tmp/ILSVRC2014/ILSVRC2014_DET_bbox_train tmp/alov/imagedata++ tmp/alov/alov300++_rectangleAnnotation_full
