#!/bin/bash

# Halt on errors
set -e

# -p also creates parent directory and suppresses error if dir exists
mkdir -p tmp/alov
cd tmp/alov

echo "ALOV: 10GB download, 20GB after unzip"

# -c restarts interrupted download, -N only retrieve if newer timestamp
wget -cN http://isis-data.science.uva.nl/alov/alov300++GT_txtFiles.zip
wget -cN http://isis-data.science.uva.nl/alov/alov300++_frames.zip

# -o overwrites existing files
unzip -o alov300++GT_txtFiles.zip
unzip -o alov300++_frames.zip

# Remove videos that overlap VOT 2014 test set
rm alov300++_rectangleAnnotation_full/01-Light/01-Light_video00016.ann
rm alov300++_rectangleAnnotation_full/01-Light/01-Light_video00022.ann
rm alov300++_rectangleAnnotation_full/01-Light/01-Light_video00023.ann
rm alov300++_rectangleAnnotation_full/02-SurfaceCover/02-SurfaceCover_video00012.ann
rm alov300++_rectangleAnnotation_full/03-Specularity/03-Specularity_video00003.ann
rm alov300++_rectangleAnnotation_full/03-Specularity/03-Specularity_video00012.ann
rm alov300++_rectangleAnnotation_full/10-LowContrast/10-LowContrast_video00013.ann
rm -r imagedata++/01-Light/01-Light_video00016
rm -r imagedata++/01-Light/01-Light_video00022
rm -r imagedata++/01-Light/01-Light_video00023
rm -r imagedata++/02-SurfaceCover/02-SurfaceCover_video00012
rm -r imagedata++/03-Specularity/03-Specularity_video00003
rm -r imagedata++/03-Specularity/03-Specularity_video00012
rm -r imagedata++/10-LowContrast/10-LowContrast_video00013

cd ../../
