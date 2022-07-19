#!/usr/bin/env sh

#Do not touch me
CAIMAN_PYTHON=/home/blansdel/anaconda3/envs/caiman/bin/python

##############
# Input data #
##############

TONE_FILE=/media/core/core_operations/ImageAnalysisScratch/Zakharenko/Jay/ROI_screening_data/newdata/roiscan1.csv

#Process unstabilized data. Path is the whole folder of tiff files
PATH_IN=/media/core/core_operations/ImageAnalysisScratch/Zakharenko/Jay/ROI_screening_data/newdata/TSeries-07062022-roiscan-001/

## Process stabilized image. Path is the specific file
#PATH_IN=/media/core/core_operations/ImageAnalysisScratch/Zakharenko/Jay/ROI_screening_data/newdata/TSeries-07062022stable-001.tif

## Explanation
# * If you provide a directory in $PATH_IN, the script will look for a file named [something]_000001.ome.tif, and load that as the first frame. Can use this for the unprocessed data.
# * If you provide a path to a specific tiff file in $PATH_IN, the script will try to treat this as the first frame in a tiff stack. This can be either a single, multi-page tiff, or the first frame of a stack of files. 
#   If the latter, this file must indeed be the first frame of that stack (having metadata that points to the rest of the files). Can use this for a stabilized image file.

################
# Output paths #
################

#Where to save results and plots to
OUTPUT_FILE=./roi_decoder_trial_run_results/roi_decoding_output_scan1.pkl
PLOT_PATH=./roi_decoder_trial_run_results/roi_decoding_output_scan1_plot

##############
# Parameters #
##############

#Grid size over which to build a decoder from. Will split the down-sampled image into sub-regions of this size, and build a separate decoder for each of these sub-regions. 
#The smaller this is the more localized the decoder, but the worse the performance may be.
ROI_SIZE=6

#Down-sample image by this factor. 
SCALE_FACTOR=16

########################################
########################################

## With non-transparent background and a sample frame
# $CAIMAN_PYTHON roi_decoder_cli.py $PATH_IN $TONE_FILE $OUTPUT_FILE \
#                                 --plotpath $PLOT_PATH \
#                                 --scalefactor $SCALE_FACTOR \
#                                 --roisize $ROI_SIZE

## With transparent background and no sample frame
# $CAIMAN_PYTHON roi_decoder_cli.py $PATH_IN $TONE_FILE $OUTPUT_FILE \
#                                 --plotpath $PLOT_PATH \
#                                 --scalefactor $SCALE_FACTOR \
#                                 --roisize $ROI_SIZE --transparent

## With transparent background and no sample frame, custom plotting range
$CAIMAN_PYTHON roi_decoder_cli.py $PATH_IN $TONE_FILE $OUTPUT_FILE \
                                --plotpath $PLOT_PATH \
                                --scalefactor $SCALE_FACTOR \
                                --roisize $ROI_SIZE --transparent \
                                --plotbounds 0.1,0.5