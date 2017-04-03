#!/bin/bash

if [ -z "$4" ]
  then
    echo "No folder supplied!"
    echo "Usage: bash `basename "$0"` imagenet_folder imagenet_annotations_folder alov_videos_folder alov_annotations_folder"
    exit
fi

set -x

GPU_ID=0
FOLDER=GOTURN1
RANDOM_SEED=800

echo FOLDER: $FOLDER

VIDEOS_FOLDER_IMAGENET=$1
ANNOTATIONS_FOLDER_IMAGENET=$2
VIDEOS_FOLDER=$3
ANNOTATIONS_FOLDER=$4
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
LAMBDA_SHIFT=5  # translation of +/-(1/5) (20% of width)
LAMBDA_SCALE=15  # scaling of +/-(1/15) (7% change in width)
MIN_SCALE=-0.4
MAX_SCALE=0.4
# According to Section G above, should use validation set for tuning
# Instead this is chosen empirically from the 3 ALOV videos with rotation.
# From manual observation, several videos do a full rotation approximately
# every second, which equates to 15 degrees (360 degrees in 24 frames).
# LAMBDA_ROTATION = 1/24 (fraction of 360 degree rotation)
# 6 - diving
# 11 - gymnastics
# 15 - motocross
# (videos 0 and 18 have rotating balls but are hard to track)
LAMBDA_ROTATION=24  # rotation of +/-(1/24) (15 degrees)

echo LAMBDA_SCALE: $LAMBDA_SCALE
echo LAMBDA_SHIFT: $LAMBDA_SHIFT
echo LAMBDA_ROTATION: $LAMBDA_ROTATION

build/train $VIDEOS_FOLDER_IMAGENET $ANNOTATIONS_FOLDER_IMAGENET $VIDEOS_FOLDER $ANNOTATIONS_FOLDER $CAFFE_MODEL $TRAIN_PROTO $SOLVER_TEMP $LAMBDA_SHIFT $LAMBDA_SCALE $MIN_SCALE $MAX_SCALE $LAMBDA_ROTATION $GPU_ID $RANDOM_SEED 2> $RESULT_DIR/results.txt

# DO NOT COMMIT
# bash scripts/train.sh tmp/ILSVRC2014/ILSVRC2014_DET_train_extracted/ tmp/ILSVRC2014/ILSVRC2014_DET_bbox_train tmp/alov/imagedata++ tmp/alov/alov300++_rectangleAnnotation_full

# build/train tmp/ILSVRC2014/ILSVRC2014_DET_train_extracted/ tmp/ILSVRC2014/ILSVRC2014_DET_bbox_train tmp/alov/imagedata++ tmp/alov/alov300++_rectangleAnnotation_full nets/models/pretrained_model/tracker.caffemodel nets/tracker.prototxt nets/solver_temp/solver_temp_GOTURN1.prototxt 5 15 -0.4 0.4 24 0 800