import skimage.io 
import numpy as np
import argparse 

parser = argparse.ArgumentParser(description='Reshape mesoscope data')
parser.add_argument('fn_in', type=str, help='Input file')
parser.add_argument('--fn_out', type=str, help='Output file')
parser.add_argument('--nblocks', type=float, default=4, help='Number of blocks the mesoscope was split into')

def main(args):

    fn_in = args.fn_in
    fn_out = args.fn_out
    n_clusters = args.nblocks - 1

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