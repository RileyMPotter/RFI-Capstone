from scipy.io import loadmat
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend, dump, load
from tqdm import tqdm

# Load data
data = loadmat("D:/WVU Classes/RFI-Capstone/Matlab Files/j1713_mat_0.mat")

# Constants
numChannels = 4096
timeSamples = 65024
sigma = 4
theta = sigma / 0.6745
scale = np.arange(0, 2 ** 16, dtype=np.float32)
v = np.arange(1, 2 ** 16 + 1, dtype=np.float32)
nSub = 512
nBlock = 127
No = 0
position = No * nBlock

reData = data["re"]
imData = data["im"]

re_chunk = reData[:, position * nSub: (position + nBlock) * nSub, 0].astype(np.float32)
im_chunk = imData[:, position * nSub: (position + nBlock) * nSub, 0].astype(np.float32)
psd_chunk = re_chunk * re_chunk + im_chunk * im_chunk

# Memory-map large data to a file
dump(psd_chunk, 'psd_chunk_memmap.pkl')
psd_chunk_memmap = load('psd_chunk_memmap.pkl', mmap_mode='r')

# Vectorized histogram calculation and KL divergence computation
chunk_size = 1024  # Process in smaller chunks to avoid memory overload
KL = np.zeros((nBlock, numChannels), dtype=np.float32)

for i in tqdm(range(0, len(scale), chunk_size), desc="Calculating KL Divergence in Chunks"):
    scale_chunk = scale[i:i + chunk_size]
    v_chunk = v[i:i + chunk_size]

    tf = psd_chunk_memmap.reshape(nBlock, nSub, numChannels)
    num, _ = np.histogram(tf, bins=scale_chunk, density=True)
    num = num / np.sum(num, axis=0, keepdims=True) + 1e-10  # Normalize and avoid division by zero

    # Mean, variance, and reference PDF (Processed in chunks)
    mu = np.mean(tf, axis=1)
    sig = np.var(tf, axis=1)
    num_ref_chunk = np.exp(-scale_chunk / np.sqrt(sig[:, :, np.newaxis])) - np.exp(
        -v_chunk / np.sqrt(sig[:, :, np.newaxis])) + 1e-10
    num_ref_chunk = num_ref_chunk[:, :, :len(num)]

    # KL Divergence Calculation for the chunk
    KL += np.sum(num_ref_chunk * np.log(num_ref_chunk / num), axis=2) + np.sum(num * np.log(num / num_ref_chunk), axis=2)

# Convert KL matrix to DataFrame for easier manipulation
KL_df = pd.DataFrame(KL)

# Mask Computation (Vectorized)
alpha = KL_df.median(axis=0)
beta = (KL_df.subtract(alpha, axis=1)).abs().median(axis=0)
mask_KL = (KL_df.subtract(alpha, axis=1).div(beta, axis=1).abs() <= theta).astype(int)

# Visualize the KL matrix using a waterfall plot
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(nBlock), np.arange(numChannels))
surf = ax.plot_surface(X, Y, KL.T, cmap='viridis')
plt.colorbar(surf)
plt.xlabel('Blocks grouped by 512 time samples')
plt.ylabel('Channels, in bins')
plt.title('Relative Entropy, pol0')
plt.savefig("Relative_Entropy_Plot.png")
plt.show()


# Apply mask to PSD data in batches (using joblib for parallel processing)
def apply_mask_batch(start_ind, end_ind, psd_chunk_memmap, mask_KL, nSub, numChannels):
    burst_batch = []
    for ind in range(start_ind, end_ind):
        burst_block = np.zeros((nSub, numChannels), dtype=np.float32)
        for chan in range(numChannels):
            burst_block[:, chan] = mask_KL.iloc[ind, chan] * psd_chunk_memmap[chan, ind * nSub:(ind + 1) * nSub]
        burst_batch.append(burst_block)
    return np.vstack(burst_batch)


batch_size = 20
with parallel_backend('loky'):
    results = Parallel(n_jobs=-1)(
        delayed(apply_mask_batch)(i, min(i + batch_size, nBlock), psd_chunk_memmap, mask_KL, nSub, numChannels)
        for i in tqdm(range(0, nBlock, batch_size), desc="Applying mask to PSD data in batches")
    )
burst = np.vstack(results)

# Define burst parameters
DM = 15.917
BW = 800 * 10 ** 6  # bandwidth [Hz]
f_c = 1500097656  # center frequency [Hz]
K = 4096  # number of channels
Ts = K / BW  # sample period [sec]
flo = 1100
fhi = 1900


# Dedisperse the burst (using joblib for parallel processing)
def dedisperse_channel(i, data, shift):
    nchan, nsamp = data.shape
    dedispersed_data = np.zeros(nsamp, dtype=np.float32)
    sh = shift[i]
    if sh > 0:
        dedispersed_data[sh:] = data[i, :-sh]
    elif sh < 0:
        dedispersed_data[:sh] = data[i, -sh:]
    else:
        dedispersed_data[:] = data[i, :]
    return dedispersed_data


nchan, nsamp = burst.T.shape
freqs = np.linspace(flo, fhi, nchan)
delay = 4.15e-3 * DM * (freqs ** -2 - fhi ** -2) / Ts
shift = np.round(delay).astype(int)

with parallel_backend('loky'):
    dedispersed_burst = Parallel(n_jobs=-1)(
        delayed(dedisperse_channel)(i, burst.T, shift) for i in tqdm(range(nchan), desc="Dedispersing burst"))
dedispersed_burst = np.array(dedispersed_burst, dtype=np.float32)

# Create a time array for high-resolution
delta = 1
time_high_res = np.arange(0, Ts * nSub - Ts * delta, Ts * delta, dtype=np.float32)

# Calculate the SNR (intensity) of the single pulse
intensity = np.sum(dedispersed_burst.astype(np.float32), axis=0)

# Convert intensity to DataFrame and write to a text file
intensity_df = pd.DataFrame({'Intensity': intensity})
np.savetxt("Intensity_J1713_Mat_0.txt", intensity)

# Print shape of burst array
print(burst.shape)
