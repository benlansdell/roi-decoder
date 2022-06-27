#!/usr/bin/env sh

#Do not touch me
CAIMAN_PYTHON=/home/blansdel/anaconda3/envs/caiman/bin/python

########################################
########################################

#Change this path to the folder you want to process.
DIR_IN=/media/core/core_operations/ImageAnalysisScratch/Zakharenko/Jay/ROI_screening_data/04052022/TSeries-04052022-roiscan-001/
TONE_FILE=/media/core/core_operations/ImageAnalysisScratch/Zakharenko/Jay/ROI_screening_data/04052022/04052022roiscan1.csv

#Where to save results and plots to
OUTPUT_FILE=./roi_decoder_trial_run_results/roi_decoding_output_scan1.pkl
PLOT_PATH=./roi_decoder_trial_run_results/roi_decoding_output_scan1_plot

#Grid size over which to build a decoder from. Will split the down-sampled image into sub-regions of this size, and build a separate decoder for each of these sub-regions. 
#The smaller this is the more localized the decoder, but the worse the performance may be.
ROI_SIZE=4

#Down-sample image by this factor. 
SCALE_FACTOR=64

########################################
########################################

$CAIMAN_PYTHON roi_decoder_online_cli.py $DIR_IN $TONE_FILE $OUTPUT_FILE \
                                --plotpath $PLOT_PATH \
                                --scalefactor $SCALE_FACTOR \
                                --roisize $ROI_SIZE