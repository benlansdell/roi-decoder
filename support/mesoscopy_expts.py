#%%
import skimage.io 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

#fn_in = '../demodata/mesoscope/file_00001_00001.tif'
fn_in = '../demodata/mesoscope/file_00002_00001.tif'
#fn_in = '../demodata/mesoscope/file_00003_00001.tif'
#fn_in = '../demodata/mesoscope/file_00004_00001.tif'

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
#Fit kmeans to cluster these points and find center of strips
#kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(min_pts.reshape(-1,1))
#centers = kmeans.cluster_centers_
#centers = np.sort(centers, axis=0)
#print(centers)

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
plt.imshow(im_reshaped[0, :, :])
# %%
