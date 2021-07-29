#!/bin/bash

#This list is named as "Panoptic Studio DB Ver 1.2"

# curPath=$(dirname "$0")
curPath=$(dirname "$0")

sequence=171204_pose1

vgaVideoNum=0
hdVideoNum=0
numKinectViews=1

# Skeleton
$curPath/getData.sh $sequence $vgaVideoNum $hdVideoNum

# Kinect RGB+Depth
$curPath/getData_kinoptic.sh $sequence $numKinectViews

# Extract Tar Files
$curPath/extractAll.sh $sequence

# Extract Kinect Imgs
cd $sequence
../kinectImgsExtractor.sh
cd ..