#!/bin/bash

set -e

mkdir -p tmp/vot2014 
cd tmp/vot2014

echo "VOT2014: 433MB download, 887MB after unzip"

wget -cN http://box.vicos.si/vot/vot2014.zip

unzip -o vot2014.zip

cd ../../
