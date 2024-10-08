import numpy as np
import matplotlib.pyploy as plt
import time
import riptide
import scipy.io

# Parameters
K = 4096  # Number of channels
N = 65024  # Number of time samples
Nsub = 512
Nblock = 127
sigma = 4
No = 0  # Varies form 0 to 4 (5 chunks in total)
Th = sigma / 0.6754  # Original value of Th is 3.5 / 0.6745

# Load the .mat file
mat_data = scipy.io.loadmat('j1713_mat_0.mat')
re = mat_data['re']
im = mat_data['im']

position = No * Nblock
re_chunk = re[:, position * Nsub: (position + Nblock) * Nsub].astype(np.float32)
im_chunk = im[:, position * Nsub: (position + Nblock) * Nsub].astype(np.float32)

# Calculate Power Specral Density (PSD)
psd_chunk = re_chunk ** 2 + im_chunk ** 2

# Histogram Bins
scale = np.arange(0, 2**16)

# Initialize KL Divergence and mask
KL = np.zeros((Nblock, K))
mask_KL = np.zeros((Nblock, K))

# Calculate KL Divergence per channel and segment
for ind in range(Nblock):
    tf = psd_chunk[:, ind * Nsub: (ind + 1) * Nsub].T

    for chan in range(K):
        # Histogram of intensity per channel per segment
        num, pos = np.histogram(tf[:, chan], bins=scale, density=True)
        num = num / np.sum(num + 1e-10)

        # Reference data: mean and variance per channel
        mu = np.mean(tf[:, chan])
        sig = np.vat(tf[:, chan])

        # Referennce pdf
        num_ref = np.exp(-scale / np.sqrt(sig)) - np.exp(-scale / np.sqrt(sig)) + 1e-10

        # Relative entropy per channel per segment
        KL[ind, chan] = np.sum(num_ref * np.log(num_ref / num)) + np.sum(num * np.log(num / num_ref))

# Compute the mask for KL distance
for chan in range(k):
    alpha = np.median(KL[:, chan])
    beta = np.median(np.abs(KL[:, chan] - alpha))

    for ind in range(Nblock):
        if abs((KL[ind, chan] - alpha) / beta) > Th:
            mask_KL[ind, chan] = 0
        else:
            mask_KL[ind, chan] = 1

# Apply mask to KL matrix and compute burst data
burst = np.zeros_like(psd_chunk)
for ind in range(Nblock):
    for chan in range(K):
        burst[(ind * Nsub):(ind + 1) * Nsub, chan] = mask_KL[ind, chan] * psd_chunk[chan, (ind * Nsub):(ind + 1) * Nsub]

# Paramteres for dedispersion
DM = 15.917
BW = 800 * 10**6    # Bandwidth [Hz]
f_c = 150009765     # Center frequency [Hz]
Ts = K / BW         # Sample period [sec]
delta = 1

# Dedisperse the burst using Riptide Library
dedispersed_burst = riptide.dedisperse(burst.T, DM, BW, f_c, K, Ts)

# Create time vector
time_high_res = np.arange(0, Ts * len(dedispersed_burst), Ts * delta, dtype=np.float32)

# Calculate intensity
intensity = np.sum(dedispersed_burst, axis=0).astype(np.float32)

# Plot the intesity
plt.figure()
plt.plot(time_high_res, np.convolve(intensity - np.mean(intensity), np.ones(32), mode='same'))
plt.xlabel('Time (s)')
plt.ylabel('Intesity')
plt.title('SNR of Signle Pulse')
plt.grid(True)
pls.show()

print(f"Execution time: {time.time()} seconds")




