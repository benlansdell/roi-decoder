#%%
import skimage.io 
import matplotlib.pyplot as plt
import numpy as np

#fn_in = '../demodata/mesoscope/file_00001_00001.tif' ## 4 blocks, set n_clusters = 3
#fn_in = '../demodata/mesoscope/file_00002_00001.tif' ## 4 blocks, set n_clusters = 3
#fn_in = '../demodata/mesoscope/file_00003_00001.tif' ## 2 blocks, set n_clusters = 1
#fn_in = '../demodata/mesoscope/file_00004_00001.tif' ## 2 blocks, set n_clusters = 1

#fn_in = '../demodata/mesoscope_w_sound/11102022_00001_00001.tif' ## 4 blocks, set n_clusters = 3
fn_in = '../demodata/mesoscope_w_sound/11102022_00007_00001.tif' ## 4 blocks, set n_clusters = 3

quant = 0.01
n_clusters = 3
#n_clusters = 1

#%%
im = skimage.io.imread(fn_in)

# %%
max_trace = np.max(im, axis=(0,2))
min_max = np.min(max_trace)
quantile = np.quantile(max_trace, quant)
min_pts = np.where(max_trace <= quantile)[0]

# %%
block_width = im.shape[2]
gap = (im.shape[1] - im.shape[2]*(n_clusters+1))/(n_clusters)
print("Gap is", gap)

# %%
strips = []
y = 0
for i in range(n_clusters):
    y += block_width 
    strips.append([y, y + gap])
    y += gap 

# %%
plt.plot(max_trace)
plt.axhline(quantile, color='r')
for (start, end) in strips:
    plt.axvline(start, color='g')
    plt.axvline(end, color='g')
#plt.plot(min_pts)
plt.xlabel('y coord')
plt.ylabel('max intensity (over x, time)')
plt.legend(['max intensity', f'{quant} quantile', 'strip boundaries'])
plt.show()

# %% Cut ouf strips and retile 
y_slices = np.ones(im.shape[1], dtype=bool)
for (start, end) in strips:
    y_slices[int(start):int(end)] = False
im_stripped = im[:, y_slices, :]

blocks = []
for i in range(n_clusters+1):
    blocks.append(im_stripped[:, i*block_width:(i+1)*block_width, :])

im_reshaped = np.concatenate(blocks, axis=2)

# %%
im_reshaped.shape
# %%
#This should be stretched??
plt.imshow(im_reshaped[0, :, :])
plt.show()

# %%
fn_out = fn_in.replace('.tif', '_reshaped.tif')
skimage.io.imsave(fn_out, im_reshaped)
# %%
