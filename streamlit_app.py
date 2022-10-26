#!/usr/bin/env python
import os
import pickle
import argparse
import time

import streamlit as st
import numpy as np

from lib import make_plots, build_localized_decoder, on_load, load_image, get_last_directories_used, save_last_directories_used
from directorypicker import st_file_selector

parser = argparse.ArgumentParser("ROI decoder")
parser.add_argument("--tifname", type=str, help="Path to the tiff file to load as a default", default = None)
parser.add_argument("--tonefile", type=str, help="Path to tone file as a default", default = None)
parser.add_argument("--pruning_dir", type=str, help="Path to the directory containing the pruning files", \
    default = '/media/core/core_operations/ImageAnalysisScratch/Zakharenko/Jay/forABBAS_BEN/For_ROI_algorithm/framelookuptable')

#Only runs on startup. Load previously selected paths
@st.cache
def first_set_directories():
    print("Setting dirs")
    last_used = get_last_directories_used()
    st.session_state['tifcurr_dir'] = last_used['tiff']
    st.session_state['soundcurr_dir'] = last_used['sound']

def force_set_directories():
    print("Setting dirs")
    last_used = get_last_directories_used()
    st.session_state['tifcurr_dir'] = last_used['tiff']
    st.session_state['soundcurr_dir'] = last_used['sound']

def st_main(args):

    def plot_button():
        os.makedirs(os.path.dirname(plot_path), exist_ok = True)
        fn_out = f'{plot_path}_{basename}_localized_decoder_boxsize_{box_size}_nframes_{n_frame:04}_f1_score.png'
        fn_out = os.path.join(os.path.dirname(st.session_state['tif_file']), fn_out)
        os.system(f"cp {tmp_path_f1} {fn_out}")
        #fig_f1.savefig(fn_out, transparent=transparent)
        fn_out = f'{plot_path}_{basename}_localized_decoder_boxsize_{box_size}_nframes_{n_frame:04}_precision.png'
        fn_out = os.path.join(os.path.dirname(st.session_state['tif_file']), fn_out)
        os.system(f"cp {tmp_path_pr} {fn_out}")
        #fig_pr.savefig(fn_out, transparent=transparent)
        fn_out = f'{plot_path}_{basename}_localized_decoder_boxsize_{box_size}_nframes_{n_frame:04}_recall.png'
        fn_out = os.path.join(os.path.dirname(st.session_state['tif_file']), fn_out)
        os.system(f"cp {tmp_path_re} {fn_out}")
        #fig_re.savefig(fn_out, transparent=transparent)

    def results_button():
        os.makedirs(os.path.dirname(out_path), exist_ok = True)
        with open(out_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Default values to load
    tone_file = args.tonefile
    tif_name = args.tifname

    if not 'tifcurr_dir' in st.session_state:
        force_set_directories()
    first_set_directories()

    start_time = time.time()

    with st.expander("Input", expanded = True):
        tif_file = st_file_selector(st, key = 'tif', label = 'Choose tif file')
        sound_file = st_file_selector(st, key = 'sound', label = 'Choose sound data')
        #use_pruned = st.checkbox('Is pruned data', value = False, help = "Check if tif file is a pruned tif file. This will try to find the pruning information in the specified directory, so that frames can be matched with sound data")
        use_pruned = False
        st.button(label='Load', on_click = lambda: on_load(tif_file, sound_file, use_pruned))

    selected_tif = st.session_state['tif_file'] if 'tif_file' in st.session_state else tif_name
    selected_sound = st.session_state['sound_file'] if 'sound_file' in st.session_state else tone_file

    #Set directories
    tiff_directory = os.path.abspath(os.path.normpath(st.session_state['tifcurr_dir']))
    sound_directory = os.path.abspath(os.path.normpath(st.session_state['soundcurr_dir']))
    save_last_directories_used(tiff_directory, sound_directory)

    if selected_tif is not None and selected_sound is not None:
        all_frames = load_image(selected_tif)
    else: 
        all_frames = None

    if all_frames is None: 
        n_frames_total = 1
    else:
        n_frames_total = len(all_frames)

    st.sidebar.title("ROI decoder")
    form = st.sidebar.form(key='my_form')
    form.write("Decoder parameters")
    n_frame = form.slider("Number of frames", 0, n_frames_total, value = n_frames_total, help = 'Number of frames to use for decoding')
    scale_factor = form.slider("Scale factor", 4, 64, value = 64, step = 8, help = 'Downsample the image by this factor')
    box_size = form.slider("Box size", 1, 10, value = 4, help = 'Size of the box to use for decoding ("pixels" in downsized image)')

    form.write("Visualisation parameters")
    transparent = form.checkbox("Transparent overlay", value = False, help = 'Overlay the decoded ROI on the original image')
    custom_range = form.checkbox("Custom color range", value = False, help = 'Use a custom range for the colorbar')
    plot_min = form.slider('V min', 0.0, 1.0, value = 0., step = 0.1)
    plot_max = form.slider('V max', 0., 1.0, value = 0.8, step = 0.1)
    if custom_range:
        if plot_max < plot_min: plot_max = plot_min
        plotbounds = [plot_min, plot_max]
    else:
        plotbounds = None
    submit = form.form_submit_button(label='Run')

    if all_frames is None: 
        st.warning('No tif file loaded')
        return

    if (submit or plot_button or results_button):
        f1_scores, re_scores, pr_scores, acc_scores, im, n_frame, overall_f1, grid_dim, all_tones = \
                        build_localized_decoder(selected_sound, all_frames, 
                                                box_size = box_size, n_frames = n_frame,
                                                scale_factor = scale_factor, im_file=selected_tif)
        if f1_scores is None:
            return
    else: 
        #Return generate empty responses 
        f1_scores = np.zeros((1,1,1))
        re_scores = np.zeros((1,1,1))
        pr_scores = np.zeros((1,1,1))
        acc_scores = np.zeros((1,1,1))
        im = np.zeros((1,1,1))
        n_frame = 1
        overall_f1 = np.zeros((1,1))
        grid_dim = 1
        all_tones = [0]

    basename = os.path.basename(selected_tif).split('.')[0]

    tmp_path_f1, tmp_path_pr, tmp_path_re = make_plots(f1_scores, pr_scores, re_scores, im, grid_dim, box_size, all_tones[1:], transparent, plotbounds)

    with st.expander("F1", expanded = True):
        st.markdown('<div style="text-align: center;">F1 scores</div>', unsafe_allow_html=True)
        st.image(tmp_path_f1)

    with st.expander("Precision"):
        st.markdown('<div style="text-align: center;">Precision scores</div>', unsafe_allow_html=True)
        st.image(tmp_path_pr)

    with st.expander("Recall"):
        st.markdown('<div style="text-align: center;">Recall scores</div>', unsafe_allow_html=True)
        st.image(tmp_path_re)

    with st.expander("Output"):
        plot_path = st.text_input('Plot path', value = './roi_decoding_results', help = 'Filename to save the plots. Will save to same directory as selected tiff file')
        out_path = st.text_input('Output file', value = './results.pkl', help = 'Filename to save the results. Will save to same directory as selected tiff file')
        st.button('Save plots', on_click= plot_button)
        st.button('Save results', on_click = results_button)

    end_time = time.time()

    results = {'n_frame': n_frame, 'box_size': box_size, 'f1': f1_scores, 're': re_scores, 'pr': pr_scores, 'acc': acc_scores,
            'overall_f1': overall_f1}

    print(f"Total time: {int(end_time-start_time)} seconds. Decoded {n_frame} frames. Overall F1 scores: {overall_f1}")

if __name__ == '__main__':
    args = parser.parse_args()
    st_main(args)

#args = parser.parse_args(['--tifname', './demodata/TSeries-07062022-001_rig__d1_512_d2_512_d3_1_order_F_frames_4000_.tif', '--tonefile', './demodata/roiscan1.csv'])
#st_main(args)

#/home/blansdel/ImageAnalysisScratch/Zakharenko/Jay/forABBAS_BEN/For_validating/DFF/frameNumber_loolup_TSeries-01062022-001-autoclip_gaussin.csv