#!/usr/bin/env python
import os 
from glob import glob 
import skimage.io
from skimage.transform import downscale_local_mean
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

from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser("ROI decoder")
parser.add_argument("tifname", type=str, help="Path to the tiff file to load")
parser.add_argument("tonefile", type=str, help="Path to tone file")
parser.add_argument("outputfile", type=str, help="Path to save decoder models and stats to")
parser.add_argument("--plotpath", type = str, default = None, help = "Filename to save decoder plot to. If not provided, then don't plot.")
parser.add_argument("--scalefactor", type=int, default = 128, help="Downsize image by this factor")
parser.add_argument("--roisize", type=int, default = 4, help="Dimension of ROI for each decoder, in units of pixels in the downsized image")
parser.add_argument("--nframes", type=int, default = None, help="Number of frames to train decoder on. If not provided, use all available")

#Where to save results and plots to
args = parser.parse_args(["/media/core/core_operations/ImageAnalysisScratch/Zakharenko/Jay/ROI_screening_data/06282022/TSeries-06282022s-002/",
                        "/media/core/core_operations/ImageAnalysisScratch/Zakharenko/Jay/ROI_screening_data/06282022/06282022roiscan2.csv",
                        "./roi_decoder_trial_run_results/roi_decoding_output_scan1.pkl",
                        "--plotpath", "./roi_decoder_trial_run_results/roi_decoding_output_scan1_plot",
                        "--scalefactor", "64",
                        "--roisize", "4"])

MLModel = LogisticRegression

#Suppress all warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def plot_decoder(plt_data, im, grid_dim = 8, title = None, box_size = 4, freq_labels = None):
    units = grid_dim*2
    demo_frame = 10
    fig, ax = plt.subplots(2,2,figsize=(16,16))
    im_dim = im.shape[1]

    max_score = np.max(plt_data[:,:,1:])

    for idx in range(4):
        i,j = idx//2, idx%2
        ax[i,j].imshow(im[demo_frame,:,:], cmap = 'gray', extent = (0, im_dim, 0, im_dim))

        axins = ax[i,j].inset_axes([(box_size-1)/units, (box_size-1)/units, (units-2*box_size+2)/units, (units-2*box_size+2)/units])
        cols = axins.imshow(plt_data[:,:,idx+1], extent = (im_dim*(box_size/units), im_dim*(1-box_size/units), im_dim*(box_size/units), im_dim*(1-box_size/units)), alpha = 0.6, vmin = 0, vmax = max_score, cmap = 'plasma')

        axins.axis('off');
        ax[i,j].axis('off');

        divider = make_axes_locatable(ax[i,j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(cols, cax = cax)
        if freq_labels is not None:
            ax[i,j].set_title(freq_labels[idx] + 'Hz')

    if title is not None:
        fig.suptitle(title)

    return fig, ax


def build_localized_decoder(tone_file, tif_name, box_size = 4, n_frames = None,
                            scale_factor = 128):

    tiff_dir = os.path.dirname(tif_name)
    print("tiff dir", tiff_dir)
    tiff_fn = glob(f'{tiff_dir}/*_000001.ome.tif')
    print("tiff fn", tiff_fn)


    from skimage.io import imread, imsave
    from joblib import Parallel, delayed

    def read_images(image_paths_list):
        images = Parallel(n_jobs=20, verbose=5)(
            delayed(lambda x: imread(x, img_num=0))(f) for f in image_paths_list
        )
        return images
        
    print('Loading', tiff_fn)
    start_time = time.time()
    if len(tiff_fn) == 0: print("Failed to find files")
    tiff_fn = tiff_fn[0]

    # Import the image
    im = skimage.io.imread(tiff_fn)

    if len(im.shape) == 2:
        #Hmm we only have the first frame, try something else to get the full stack:
        reader = TiffGlobReader(f'{tiff_dir}/*.tif', indexer = lambda x: pd.Series({'T':int(os.path.basename(x).split('_')[-1].split('.')[0])}))
        im = np.squeeze(reader.data)

    im = im[im.sum(axis = 1).sum(axis = 1) > 0,:,:]

    end_time = time.time()
    print(f"{int(end_time - start_time)} seconds to load tiff stack.")

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
        im_downscaled = np.zeros((im.shape[0], im.shape[1]//scale_factor, im.shape[2]//scale_factor))
    except IndexError:
        print("Couldn't load full tiff stack. Exiting")
        return 

    grid_dim = im_downscaled.shape[1]

    for idx in range(im.shape[0]):
        im_downscaled[idx,:,:] = downscale_local_mean(im[idx,:,:], (scale_factor, scale_factor))

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

    for i in tqdm(range(n_grid_pts)):
        f1_row = []
        re_row = []
        pr_row = []
        acc_row = []
        for j in range(n_grid_pts):
            data = grid_downscaled[:,i:(i+box_size),j:(j+box_size)].reshape((-1, box_size*box_size))
            model = MLModel()
            splitter = KFold(n_splits=5, shuffle = False)
            pred_prob_tones = cross_val_predict(model, data, tone_labels, cv = splitter)
            recall = recall_score(tone_labels, pred_prob_tones, average = None, labels = all_tones)
            prec = precision_score(tone_labels, pred_prob_tones, average = None, labels = all_tones)
            f1 = f1_score(tone_labels, pred_prob_tones, average = None, labels = all_tones)
            acc = accuracy_score(tone_labels, pred_prob_tones)

            f1_row.append(f1)
            pr_row.append(prec)
            re_row.append(recall)
            acc_row.append(acc)

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
    overall_f1

    return f1_scores, re_scores, pr_scores, acc_scores, im, n_frames, overall_f1, grid_dim, all_tones

def main(args):

    tone_file = args.tonefile
    tif_name = args.tifname
    plot_path = args.plotpath
    out_path = args.outputfile

    scale_factor = args.scalefactor
    box_size = args.roisize
    n_frame = args.nframes

    start_time = time.time()
    f1_scores, re_scores, pr_scores, acc_scores, im, n_frame, overall_f1, grid_dim, all_tones = \
                    build_localized_decoder(tone_file, tif_name,
                                            box_size = box_size, n_frames = n_frame,
                                            scale_factor = scale_factor)

    basename = os.path.basename(tif_name).split('.')[0]

    #Get base dir names 
    os.makedirs(os.path.dirname(out_path), exist_ok = True)
    os.makedirs(os.path.dirname(plot_path), exist_ok = True)

    if plot_path is not None:
        fig, _ = plot_decoder(f1_scores, im, grid_dim = grid_dim, title = "F1 scores", box_size = box_size, freq_labels = all_tones[1:])
        fn_out = f'{plot_path}_{basename}_localized_decoder_boxsize_{box_size}_nframes_{n_frame:04}_f1_score.png'
        fig.savefig(fn_out)

        fig, _ = plot_decoder(pr_scores, im, grid_dim = grid_dim, title = "Precision scores", box_size = box_size, freq_labels = all_tones[1:])
        fn_out = f'{plot_path}_{basename}_localized_decoder_boxsize_{box_size}_nframes_{n_frame:04}_precision.png'
        fig.savefig(fn_out)

        fig, _ = plot_decoder(re_scores, im, grid_dim = grid_dim, title = "Recall scores", box_size = box_size, freq_labels = all_tones[1:])
        fn_out = f'{plot_path}_{basename}_localized_decoder_boxsize_{box_size}_nframes_{n_frame:04}_recall.png'
        fig.savefig(fn_out)

    end_time = time.time()

    results = {'n_frame': n_frame, 'box_size': box_size, 'f1': f1_scores, 're': re_scores, 'pr': pr_scores, 'acc': acc_scores,
               'overall_f1': overall_f1}

    print(f"Total time: {int(end_time-start_time)} seconds. Decoded {n_frame} frames. Overall F1 scores: {overall_f1}")

    with open(out_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
