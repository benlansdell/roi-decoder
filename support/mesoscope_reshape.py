"""
Note this takes the blocks, removes the strips and aligns them to get back the original image. 

It currently does not stretch the resulting image to get a square. Such reshaping would add no extra information
for a decoder to use, so it is not done here.

The decoder should take into account the different sizes blocks.
"""

import skimage.io 
import numpy as np
import argparse 

parser = argparse.ArgumentParser(description='Reshape mesoscope data, removing strips that were added for some reason')
parser.add_argument('fn_in', type=str, help='Input file')
parser.add_argument('--fn_out', type=str, default = None, help='Output file. If not provided, will be the input file with _reshaped.tif appended')
parser.add_argument('--n_blocks', type=int, default=4, help='Number of blocks the mesoscope was split into')

def main(args):

    fn_in = args.fn_in
    fn_out = args.fn_out
    n_clusters = args.n_blocks - 1

    im = skimage.io.imread(fn_in)
    block_width = im.shape[2]
    gap = (im.shape[1] - im.shape[2]*(n_clusters+1))/(n_clusters)
    print("Gap is", gap)

    strips = []
    y = 0
    for i in range(n_clusters):
        y += block_width 
        strips.append([y, y + gap])
        y += gap 

    y_slices = np.ones(im.shape[1], dtype=bool)
    for (start, end) in strips:
        y_slices[int(start):int(end)] = False
    im_stripped = im[:, y_slices, :]

    blocks = []
    for i in range(n_clusters+1):
        blocks.append(im_stripped[:, i*block_width:(i+1)*block_width, :])
    im_reshaped = np.concatenate(blocks, axis=2)

    if fn_out is None:
        fn_out = fn_in[:-4] + "_reshaped.tif"
    skimage.io.imsave(fn_out, im_reshaped)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)