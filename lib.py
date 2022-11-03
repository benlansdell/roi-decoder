#!/usr/bin/env python
import os
import skimage.io
import time
import warnings
import json
import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from tqdm import tqdm
from glob import glob 

from skimage.transform import downscale_local_mean
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression

from aicsimageio.readers import TiffGlobReader
from mpl_toolkits.axes_grid1 import make_axes_locatable


MLModel = LogisticRegression
MASK_BELOW = 0.01

last_directories_used_file = 'last_directory_used.json'
def get_last_directories_used():
    #Load json file last_directory_used_file to directories
    if os.path.exists(last_directories_used_file):
        with open(last_directories_used_file, 'r') as f:
            directories = json.load(f)
    else:
        directories = {'tiff': '.', 'sound': '.'}
    return directories

def save_last_directories_used(tiff_directory, sound_directory):
    d = {'tiff': tiff_directory,
         'sound': sound_directory}
    if os.path.exists(last_directories_used_file):
        os.remove(last_directories_used_file)

    json_object = json.dumps(d, indent=4)
    with open(last_directories_used_file, "w") as outfile:
        outfile.write(json_object)

#Suppress all warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def plot_decoder(plt_data, im, grid_dim = 8, title = None, box_size = 4, freq_labels = None, transparent = False, plotbounds = None):
    units = grid_dim*2

    n_freqs = plt_data.shape[2]-1
    n_plots_y = n_freqs//2

    demo_frame = 10
    fig, ax = plt.subplots(n_plots_y, 2,figsize=(16,8*n_plots_y))
    im_dim = im.shape[1]

    max_score = np.max(plt_data[:,:,1:])

    if plotbounds is None:
        mask_below = MASK_BELOW
        bounds = (0, max_score)
    else:
        mask_below = plotbounds[0]
        bounds = plotbounds

    for idx in range(n_freqs):
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

@st.cache(suppress_st_warning=True)
def load_image(tif_name : str) -> np.ndarray:
    st.spinner(text="Loading image stack")

    start_time = time.time()
    ##Read a directory of tiff files
    if os.path.isdir(tif_name):
        tiff_dir = os.path.dirname(tif_name)
        tiff_fn = glob(f'{tiff_dir}/*_000001.ome.tif')

        print('Loading', tiff_fn)
        if len(tiff_fn) == 0:
            raise ValueError("Failed to find any file matching *_000001.ome.tif in provided directory")
        tiff_fn = tiff_fn[0]
    ##Read a single tiff file
    else:
        tiff_fn = tif_name
        tiff_dir = os.path.dirname(tiff_fn)

    # Import the image
    try:
        im = skimage.io.imread(tiff_fn)

        #Hmm we only have the first frame, try something else to get the full stack:
        if len(im.shape) == 2:
            print("Only loaded one frame, attempting to load the rest with TiffGlobReader. For this to work, {tiff_dir}/*.tif must correspond to the files you want to process.")
            reader = TiffGlobReader(f'{tiff_dir}/*.tif', indexer = lambda x: pd.Series({'T':int(os.path.basename(x).split('_')[-1].split('.')[0])}))
            im = np.squeeze(reader.data)
        im = im[im.sum(axis = 1).sum(axis = 1) > 0,:,:]

    except:
        st.warning("Failed to load image stack. Please check that the file is a valid tiff stack.")
        im = None

    end_time = time.time()
    print(f"{int(end_time - start_time)} seconds to load tiff stack.")
    return im

@st.cache(suppress_st_warning=True)
def build_localized_decoder(tone_file, im, box_size = 4, n_frames = None,
                            scale_factor = 128, use_pruned = False, im_file = None,
                            prune_dir = None):

    empty_val = [None]*9

    if n_frames is not None and not use_pruned:
        im = im[:n_frames,:,:]
    else:
        n_frames = im.shape[0]
                
    try:
        tones = pd.read_csv(tone_file, header = None, names = ['time', 'freq', 'atten'])
    except:
        try:
            tones = pd.read_csv(tone_file)
            tones.columns = ['time', 'freq', 'atten']
        except:
            st.warning("Failed to load tone file. Please check file is a valid csv file.")
            return empty_val

    if use_pruned:
        #Compute tone vector, and only grab frames where are in OLDframe column in pruning file
        pass
        #match: frameNumber_loolup_TSeries-01062022-001-autoclip_gaussin.csv
        #to: TSeries-07062022-001_rig__d1_512_d2_512_d3_1_order_F_frames_4000_.tif
        im_file_stub = 'TSeries' + '_'.join(os.path.basename(im_file).split('TSeries')[1].split('_')[:1])
        im_file_stub = im_file_stub.replace('-autoclip.tif', '')
        prune_file = glob(f'{prune_dir}/*{im_file_stub}*.csv')
        if len(prune_file) == 0:
            st.warning("Failed to find a matching pruning file. Please check that the pruning directory is correct.")
            return empty_val
        if len(prune_file) > 1:
            st.warning("Found multiple matching pruning files. Please check that the pruning directory is correct.")
            return empty_val
        prune_file = prune_file[0]

        #load in pruned frame data 
        prune_data = pd.read_csv(prune_file)

        #Prune both the tone vector and the image stack to be the same length, specified by this prune data 
        n_frames_unpruned = prune_data['OLDframe'].astype(int).max()

        ##Fill in im
        im_unpruned = np.empty((n_frames_unpruned, im.shape[1], im.shape[2]))
        im_unpruned[:] = np.nan

        for idx, row in prune_data.iterrows():
            if row['OLDframe'] <= im_unpruned.shape[0] and row['NEWframe'] <= im.shape[0]:
                im_unpruned[row['OLDframe']-1,:,:] = im[row['NEWframe']-1,:,:]

        im = im_unpruned

        tones = tones[tones['time'] < (n_frames_unpruned/10)]
    else:
        print(tones['time'])
        print(n_frames)
        tones = tones[tones['time'] < (n_frames/10)]

    all_tones = ['0.0'] + [str(x) for x in sorted([int(x) for x in list(tones['freq'].unique())])]
    print('Detected tones', all_tones)

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

    if use_pruned:
        #Drop nans
        tone_labels = tone_labels[~np.isnan(grid_downscaled).any(axis=(1, 2))]
        grid_downscaled = grid_downscaled[~np.isnan(grid_downscaled).any(axis=(1, 2))]

    f1_scores = []
    re_scores = []
    pr_scores = []
    acc_scores = []

    print("Fitting the ML models")

    ## Box size calcs
    n_grid_pts = grid_dim+1-box_size

    unique_tones = np.unique(tone_labels)
    unique_tones = sorted(unique_tones[unique_tones != '0.0'], key = lambda x: int(float(x)))

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
            model = MLModel()

            #Separate decoder for each tone
            f1_col = [0]
            re_col = [0]
            pr_col = [0]
            acc_col = [0]

            for idx in range(len(unique_tones)):
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

        f1_scores.append(f1_row)
        re_scores.append(re_row)
        pr_scores.append(pr_row)
        acc_scores.append(acc_row)
        
    f1_scores = np.array(f1_scores)
    re_scores = np.array(re_scores)
    pr_scores = np.array(pr_scores)
    acc_scores = np.array(acc_scores)

    model = MLModel()
    pred_prob_all = cross_val_predict(model, im_downscaled_, tone_labels, cv = splitter)
    overall_f1 = f1_score(tone_labels, pred_prob_all, average = None, labels = all_tones)
   
    return f1_scores, re_scores, pr_scores, acc_scores, im, n_frames, overall_f1, grid_dim, all_tones

#The problem with caching this is that it doesn't replot the same settings twice
#@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
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

    tmp_path = 'tmp_plots'
    tmp_f1_path = os.path.join(tmp_path, 'f1.png')
    tmp_pr_path = os.path.join(tmp_path, 'pr.png')
    tmp_re_path = os.path.join(tmp_path, 're.png')
    os.makedirs(tmp_path, exist_ok = True)
    fig_f1.savefig(tmp_f1_path)
    fig_pr.savefig(tmp_pr_path)
    fig_re.savefig(tmp_re_path)

    return tmp_f1_path, tmp_pr_path, tmp_re_path

def on_load(tif_file, sound_file, use_pruned):
    st.session_state['use_pruned'] = use_pruned
    if os.path.exists(tif_file) and os.path.exists(sound_file):
        st.session_state['tif_file'] = tif_file
        st.session_state['sound_file'] = sound_file
    if not os.path.exists(tif_file):
        st.warning('Tif file not found')
    if not os.path.exists(sound_file):
        st.warning('Sound file not found')