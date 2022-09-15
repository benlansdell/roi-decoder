#!/usr/bin/env python
import os
from glob import glob
from turtle import onclick 
import skimage.io
from skimage.transform import downscale_local_mean, rescale, resize
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse
import time

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from aicsimageio.readers import TiffGlobReader
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.io import imread, imsave
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression

import streamlit as st
from directorypicker import st_directory_picker

##Now all of these things are settings in the GUI (left panel)

parser = argparse.ArgumentParser("ROI decoder")
parser.add_argument("tifname", type=str, help="Path to the tiff file to load")
parser.add_argument("tonefile", type=str, help="Path to tone file")

#Where to save results and plots to
## Test paths for stabilized images
args = parser.parse_args(["./demo_data/TSeries-07062022-001_rig__d1_512_d2_512_d3_1_order_F_frames_4000_.tif", "./demo_data/roiscan1.csv"])

MLModel = LogisticRegression

#Suppress all warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

MASK_BELOW = 0.01

def plot_decoder(plt_data, im, grid_dim = 8, title = None, box_size = 4, freq_labels = None, transparent = False, plotbounds = None):
    units = grid_dim*2
    demo_frame = 10
    fig, ax = plt.subplots(2,2,figsize=(16,16))
    im_dim = im.shape[1]

    max_score = np.max(plt_data[:,:,1:])

    if plotbounds is None:
        mask_below = MASK_BELOW
        bounds = (0, max_score)
    else:
        mask_below = plotbounds[0]
        bounds = plotbounds

    for idx in range(4):
        i,j = idx//2, idx%2

        #Draw a sample frame
        if transparent == False:
            ax[i,j].imshow(im[demo_frame,:,:], cmap = 'gray', extent = (0, im_dim, 0, im_dim))
        else:
            #Draw blank space
            ax[i,j].imshow(np.zeros((*im.shape[1:], 4)), extent = (0, im_dim, 0, im_dim))

        axins = ax[i,j].inset_axes([(box_size-1)/units, (box_size-1)/units, (units-2*box_size+2)/units, (units-2*box_size+2)/units])

        masked_data = np.ma.masked_where(plt_data[:,:,idx+1] < mask_below, plt_data[:,:,idx+1])

        cols = axins.imshow(masked_data, 
                            extent = (im_dim*(box_size/units), im_dim*(1-box_size/units), im_dim*(box_size/units), im_dim*(1-box_size/units)), 
                            alpha = 0.6, 
                            vmin = bounds[0], 
                            vmax = bounds[1], 
                            cmap = 'plasma')

        axins.axis('off');

        if not transparent:
            ax[i,j].axis('off');
        else:
            ax[i,j].axis('on');
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].set_xticklabels([])
            ax[i,j].set_yticklabels([])

        divider = make_axes_locatable(ax[i,j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(cols, cax = cax)
        if freq_labels is not None:
            ax[i,j].set_title(freq_labels[idx] + 'Hz')

    if title is not None:
        fig.suptitle(title)

    return fig, ax

@st.cache
def load_image(tif_name : str) -> np.ndarray:
    st.spinner(text="Loading image stack")
    def read_images(image_paths_list):
        images = Parallel(n_jobs=20, verbose=5)(
            delayed(lambda x: imread(x, img_num=0))(f) for f in image_paths_list
        )
        return images

    start_time = time.time()
    ##Read a directory of tiff files
    if os.path.isdir(tif_name):
        tiff_dir = os.path.dirname(tif_name)
        print("tiff dir", tiff_dir)
        tiff_fn = glob(f'{tiff_dir}/*_000001.ome.tif')
        print("tiff fn", tiff_fn)

        print('Loading', tiff_fn)
        if len(tiff_fn) == 0:
            raise ValueError("Failed to find any file matching *_000001.ome.tif in provided directory")
        tiff_fn = tiff_fn[0]
    ##Read a single tiff file
    else:
        tiff_fn = tif_name
        tiff_dir = os.path.dirname(tiff_fn)
        print("tiff dir", tiff_dir)
        print("tiff fn", tiff_fn)

    # Import the image
    im = skimage.io.imread(tiff_fn)

    #Hmm we only have the first frame, try something else to get the full stack:
    if len(im.shape) == 2:
        print("Only loaded one frame, attempting to load the rest with TiffGlobReader. For this to work, {tiff_dir}/*.tif must correspond to the files you want to process.")
        reader = TiffGlobReader(f'{tiff_dir}/*.tif', indexer = lambda x: pd.Series({'T':int(os.path.basename(x).split('_')[-1].split('.')[0])}))
        im = np.squeeze(reader.data)

    im = im[im.sum(axis = 1).sum(axis = 1) > 0,:,:]

    end_time = time.time()
    print(f"{int(end_time - start_time)} seconds to load tiff stack.")
    return im

@st.cache(suppress_st_warning=True)
def build_localized_decoder(tone_file, im, box_size = 4, n_frames = None,
                            scale_factor = 128):

    if n_frames is not None:
        im = im[:n_frames,:,:]
    else:
        n_frames = im.shape[0]
        
    #Translate the image by half the frame
    #for idx in range(n_frames):
    #    im[idx,:,:] = np.roll(im[idx,:,:], -im.shape[1]//2, axis = 1)
    
    tones = pd.read_csv(tone_file, header = None, names = ['time', 'freq', 'atten'])
    tones = tones[tones['time'] < (n_frames/10)]

    all_tones = ['0.0'] + [str(x) for x in sorted([int(x) for x in list(tones['freq'].unique())])]

    # Do the coarse graining
    try:
        test_downscale = downscale_local_mean(im[0,:,:], (scale_factor, scale_factor))
        im_downscaled = np.zeros((im.shape[0], *test_downscale.shape))
    except IndexError:
        IndexError("Couldn't load full tiff stack. Exiting")

    grid_dim = im_downscaled.shape[1]

    for idx in range(im.shape[0]):

        im_downscaled[idx,:,:] = downscale_local_mean(im[idx,:,:], (scale_factor, scale_factor))
        #This is toooo slow
        #im_downscaled[idx,:,:] = resize(im[idx,:,:], im_downscaled.shape[1:])

    # Difference the data
    im_downscaled_ = np.reshape(im_downscaled, (im_downscaled.shape[0], -1))
    im_downscaled_ = np.diff(im_downscaled_, axis = 0)

    cue_indices = 4 + np.arange(0, im_downscaled_.shape[0], 10)
    cue_indices = cue_indices[cue_indices < im_downscaled_.shape[0]]
    labels = np.zeros(im_downscaled_.shape[0])
    labels[cue_indices] = 1

    tones['freq'] = tones['freq'].astype(str)
    tone_labels = labels.astype(str)

    tone_labels = tone_labels[:len(labels)]
    tone_labels[labels == 1] = tones['freq'][:len(labels[labels == 1])]

    grid_downscaled = im_downscaled_.reshape((-1, grid_dim, grid_dim))

    f1_scores = []
    re_scores = []
    pr_scores = []
    acc_scores = []

    print("Fitting the ML models")

    ## Box size calcs
    n_grid_pts = grid_dim+1-box_size

    unique_tones = np.unique(tone_labels)
    unique_tones = sorted(unique_tones[unique_tones != '0.0'], key = lambda x: int(x))

    prog_bar = st.sidebar.progress(0)

    for i in tqdm(range(n_grid_pts)):
        prog_bar.progress(float((i+1)/n_grid_pts))
        f1_row = []
        re_row = []
        pr_row = []
        acc_row = []
        for j in range(n_grid_pts):
            data = grid_downscaled[:,i:(i+box_size),j:(j+box_size)].reshape((-1, box_size*box_size))
            splitter = KFold(n_splits=5, shuffle = False)
            #model = MLModel(class_weight='balanced')
            model = MLModel()

            #Separate decoder for each tone
            f1_col = [0]
            re_col = [0]
            pr_col = [0]
            acc_col = [0]

            for idx in range(len(unique_tones)):
                #model = MLModel(class_weight = 'balanced')
                model = MLModel()
                tone_labels_i = (tone_labels == unique_tones[idx])
                pred_prob_tones = cross_val_predict(model, data, tone_labels_i, cv = splitter)
                recall = recall_score(tone_labels_i, pred_prob_tones)
                prec = precision_score(tone_labels_i, pred_prob_tones)
                f1 = f1_score(tone_labels_i, pred_prob_tones)
                acc = accuracy_score(tone_labels_i, pred_prob_tones)
                f1_col.append(f1)
                re_col.append(recall)
                pr_col.append(prec)
                acc_col.append(acc)

            f1_row.append(f1_col)
            pr_row.append(pr_col)
            re_row.append(re_col)
            acc_row.append(acc_col)

            # # One decoder for all tones
            # pred_prob_tones = cross_val_predict(model, data, tone_labels, cv = splitter)
            # recall = recall_score(tone_labels, pred_prob_tones, average = None, labels = all_tones)
            # prec = precision_score(tone_labels, pred_prob_tones, average = None, labels = all_tones)
            # f1 = f1_score(tone_labels, pred_prob_tones, average = None, labels = all_tones)
            # acc = accuracy_score(tone_labels, pred_prob_tones)
            # f1_row.append(f1)
            # pr_row.append(prec)
            # re_row.append(recall)
            # acc_row.append(acc)

        f1_scores.append(f1_row)
        re_scores.append(re_row)
        pr_scores.append(pr_row)
        acc_scores.append(acc_row)
        
    f1_scores = np.array(f1_scores)
    re_scores = np.array(re_scores)
    pr_scores = np.array(pr_scores)
    acc_scores = np.array(acc_scores)

    model = MLModel() #LogisticRegression()
    pred_prob_all = cross_val_predict(model, im_downscaled_, tone_labels, cv = splitter)
    overall_f1 = f1_score(tone_labels, pred_prob_all, average = None, labels = all_tones)
    
    # pred_prob_all = cross_val_predict(model, im_downscaled_, labels, cv = splitter)
    # overall_f1 = f1_score(labels, pred_prob_all)
    # overall_f1

    return f1_scores, re_scores, pr_scores, acc_scores, im, n_frames, overall_f1, grid_dim, all_tones

import matplotlib
@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def make_plots(f1_scores, pr_scores, re_scores, im, grid_dim, box_size, all_tones, transparent, plotbounds):
    fig_f1, _ = plot_decoder(f1_scores,
                        im,
                        grid_dim = grid_dim,
                        box_size = box_size,
                        freq_labels = all_tones,
                        transparent = transparent,
                        plotbounds = plotbounds)
    fig_pr, _ = plot_decoder(pr_scores,
                        im,
                        grid_dim = grid_dim,
                        box_size = box_size,
                        freq_labels = all_tones,
                        transparent = transparent,
                        plotbounds = plotbounds)
    fig_re, _ = plot_decoder(re_scores,
                        im,
                        grid_dim = grid_dim,
                        box_size = box_size,
                        freq_labels = all_tones,
                        transparent = transparent,
                        plotbounds = plotbounds)
    return fig_f1, fig_pr, fig_re

def st_main(args):

    def plot_button():
        os.makedirs(os.path.dirname(plot_path), exist_ok = True)
        fn_out = f'{plot_path}_{basename}_localized_decoder_boxsize_{box_size}_nframes_{n_frame:04}_f1_score.png'
        fig_f1.savefig(fn_out, transparent=transparent)
        fn_out = f'{plot_path}_{basename}_localized_decoder_boxsize_{box_size}_nframes_{n_frame:04}_precision.png'
        fig_pr.savefig(fn_out, transparent=transparent)
        fn_out = f'{plot_path}_{basename}_localized_decoder_boxsize_{box_size}_nframes_{n_frame:04}_recall.png'
        fig_re.savefig(fn_out, transparent=transparent)
    
    def results_button():
        os.makedirs(os.path.dirname(out_path), exist_ok = True)
        with open(out_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    tone_file = args.tonefile
    tif_name = args.tifname

    start_time = time.time()
    all_frames = load_image(tif_name)
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

    if submit or plot_button or results_button:
        f1_scores, re_scores, pr_scores, acc_scores, im, n_frame, overall_f1, grid_dim, all_tones = \
                        build_localized_decoder(tone_file, all_frames, 
                                                box_size = box_size, n_frames = n_frame,
                                                scale_factor = scale_factor)
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

    basename = os.path.basename(tif_name).split('.')[0]

    fig_f1, fig_pr, fig_re = make_plots(f1_scores, pr_scores, re_scores, im, grid_dim, box_size, all_tones[1:], transparent, plotbounds)

    with st.expander("Input", expanded = True):
        input_form = st.form(key='inputform')
        st_directory_picker(base = input_form)
        load_d = input_form.form_submit_button(label='Load')

    with st.expander("F1", expanded = True):
        st.markdown('<div style="text-align: center;">F1 scores</div>', unsafe_allow_html=True)
        st.write(fig_f1)

    with st.expander("Precision"):
        st.markdown('<div style="text-align: center;">Precision scores</div>', unsafe_allow_html=True)
        st.write(fig_pr)

    with st.expander("Recall"):
        st.markdown('<div style="text-align: center;">Recall scores</div>', unsafe_allow_html=True)
        st.write(fig_re)

    with st.expander("Output"):
        plot_path = st.text_input('Plot path', value = './roi_decoding_results', help = 'Directory to save the plots')
        out_path = st.text_input('Output file', value = './results.pkl', help = 'Path to save the results')
        st.button('Save plots', on_click= plot_button)
        st.button('Save results', on_click = results_button)

    end_time = time.time()

    results = {'n_frame': n_frame, 'box_size': box_size, 'f1': f1_scores, 're': re_scores, 'pr': pr_scores, 'acc': acc_scores,
            'overall_f1': overall_f1}

    print(f"Total time: {int(end_time-start_time)} seconds. Decoded {n_frame} frames. Overall F1 scores: {overall_f1}")


if __name__ == '__main__':
    args = parser.parse_args()
    st_main(args)