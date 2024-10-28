# Public Packages
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

# Our Packages
from astrofunc import Dedisperse

# Load data
data = loadmat("./Matlab Files/j1713_mat_0.mat")

# Constants
numChannels = 4096
timeSamples = 65024
sigma = 4
theta = sigma / 0.6745
scale = np.arange(0, 2 ** 16)
v = np.arange(1, 2 ** 16 + 1)
nSub = 512
nBlock = 127
No = 0
position = No * nBlock

# Extract real and imaginary data
reData = data["re"]
imData = data["im"]

re_chunk = reData[:, position * nSub: (position + nBlock) * nSub, 0].astype(np.float32)
im_chunk = imData[:, position * nSub: (position + nBlock) * nSub, 0].astype(np.float32)
psd_chunk = re_chunk * re_chunk + im_chunk * im_chunk

# Vectorized histogram calculation and KL divergence computation
tf = psd_chunk.reshape(nBlock, nSub, numChannels)
num, _ = np.histogram(tf, bins=scale, density=True)
num = num / np.sum(num, axis=0, keepdims=True) + 1e-10  # Normalize and avoid division by zero

# Mean, variance, and reference PDF (Vectorized)
mu = np.mean(tf, axis=1)
sig = np.var(tf, axis=1)
num_ref = np.exp(-scale / np.sqrt(sig[:, :, np.newaxis])) - np.exp(-v / np.sqrt(sig[:, :, np.newaxis])) + 1e-10
num_ref = num_ref[:, :, :len(num)]

# KL Divergence Calculation (Vectorized)
KL = np.sum(num_ref * np.log(num_ref / num), axis=2) + np.sum(num * np.log(num / num_ref), axis=2)

# Convery KL matrix to DataFrame for easier manipulation
KL_df = pd.DataFrame(KL)

# Mask Computation (Vectorized)
alpha = np.median(KL, axis=0)
beta = np.median(np.abs(KL - alpha), axis=0)
mask_KL = (np.abs((KL - alpha) / beta) <= theta).astype(int)

# Visualize thee KL matric using a waterfall plot
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(nBlock), np.arange(numChannels))
surf = ax.plot_surface(X, Y, KL.T, cmap='viridis')
plt.colorbar(surf)
plt.xlabel('Blocks grouped by 512 time samples')
plt.ylabel('Channels, in bins')
plt.title('Relative Entropy, pol0')
plt.show()

# Apply mask to PSD data (using tqdm for progress tracking, joblib for parallel processing)
def apply_mak(ind, psd_chunk, mask_KL, nSub, numChannels):
    burst_block = np.zeros((nSub, numChannels))
    for chan in range(numChannels):
        burst_block[:, chan] = mask_KL[ind, chan] * psd_chunk[chan, ind * nSub:(ind + 1) * nSub]
    return burst_block

result = Parallel(n_jobs=-1)(delayed(apply_mask)(ind, psd_chunk, mask-KL, nSub, numChannels)) for ind in tqdm(range(nBlock), desc="Applying mask to PSD data")
burst = np.vstack(results)

# Define burst parameters
DM = 15.917
BW = 800 * 10 ** 6  # bandwidth [Hz]
f_c = 1500097656    # center frequency [Hz]
K = 4096            # number of channels
Ts = K / BW         # sample period [sec]
flo = 1100
fhi = 1900

# Dedisperse the burst (using joblib for parallel processing)
def dedisperse_channel(i, data, shift):
    nchan, nsamp = data.shape
    dedispersed_data = np.zeros(nsamp)
    sh = shift[i]
    if sh > 0:
        dedispersed_data[sh:] = data[i, :-sh]
    elif sh < 0:
        dedispersed_data[:sh] = data[i, -sh:]
    else:
        dedispersed_data[:] = data[i, :]
    return dedispersed_data

nchan, nsamp, = burst.T.shape
freqs = np.linspace(flo, fhi, nchan)
delay = 4.15e-3 * DM * (freqs ** -2 - fhi ** -2) / Ts
shift = np.round(delay).astype(int)

dedispersed_burst = Parallel(n_jobs=-1)(delayed(dedisperse_channel)(i, burst.T, shift) for i in tqdm(range(nchan), desc="Dedispersing burst"))
dedispersed_burst = np.array(dedispersed_burst)

# Create a time array for high-resolution
delta = 1
time_high_res = np.arange(0, Ts * nSub - Ts * delta, Ts * delta, dtype=np.float32)

# Calculate the SNR (intensity) of the signle pulse
intensity = np.sum(dedispersed_burst.astype(np.float32), axis=0)

# Convert intensity to DataFrame and write to a txt file
intensity_df = pd.DataFrame({'Intensity': intensity})
np.savetxt("Intensity_J1713_Mat_0.txt", intensity)

# Print shape of burst array
print(burst.shape)