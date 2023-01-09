#!/bin/sh
#
# NOTE:
# Edit these three parameters as needed. Do not put a space on either side of the equals sign.
# N_BLOCKS is the number of segments captured by the mesoscope that need to be rearranged.
#
# Note this takes the blocks, removes the strips and aligns them to get back the original image.
#
# It currently does not stretch the resulting image to get a square. Such reshaping would add no extra information
# for a decoder to use, so it is not done here.
#
# The decoder should take into account the different sizes blocks.

PYTHON=/home/blansdel/anaconda3/envs/caiman/bin/python
FILE_IN="/media/ResearchHome/zakhagrp/projects/JayImagingData/common/ROI_GUI/for Ben/11102022_00007_00001.tif"
FILE_OUT="/media/ResearchHome/zakhagrp/projects/JayImagingData/common/ROI_GUI/for Ben/11102022_00007_00001_reshaped.tif"
N_BLOCKS=4

$PYTHON ./mesoscopy_reshape.py $FILE_IN $FILE_OUT $N_BLOCKS
