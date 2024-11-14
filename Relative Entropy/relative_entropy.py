from scipy.io import loadmat
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data = loadmat("./Matlab Files/j1713_mat_0.mat")

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
reData = data["re"]
imData = data["im"]
re_chunk = reData[:, position * nSub: (position + nBlock) * nSub, 0].astype(np.float32)
im_chunk = imData[:, position * nSub: (position + nBlock) * nSub, 0].astype(np.float32)

psd_chunk = re_chunk * re_chunk + im_chunk * im_chunk

# Unload mat file, clear re_chunk, and psd_chunk

# Parameters used to fit a curve in the histogram of intesnsities per channel
# scale = 0:1:2^16-1
# v = 1:2^16


count = 1
KL = np.zeros((nBlock, numChannels))

for ind in range(nBlock):  # Loop over blocks
    # Extract the block of data
    tf = psd_chunk[:, ind * nSub:(ind + 1) * nSub].T

    for chan in range(numChannels):  # Loop over channels
        # Histogram of intensity per channel per segment
        num, pos = np.histogram(tf[:, chan], bins=scale, density=True)
        num = num / np.sum(num) + 10 ** (-10)  # Normalize and avoid division by zero

        # Mean and variance per channel
        mu = np.mean(tf[:, chan])
        sig = np.var(tf[:, chan])

        # Reference PDF
        num_ref = np.exp(-scale / np.sqrt(sig)) - np.exp(-v / np.sqrt(sig)) + 10 ** (-10)

        num_ref = num_ref[:len(num)]
        # Relative entropy (KL divergence) per channel per segment
        KL[ind, chan] = np.sum(num_ref * np.log(num_ref / num)) + np.sum(num * np.log(num / num_ref))
        #      ^  Flip these? 


        # Optional plotting (if needed, uncomment the next lines)
        # plt.figure(1)
        # plt.plot(pos[:-1], num, 'b', label='Observed')
        # plt.plot(pos[:-1], num_ref, 'r', label='Reference')
        # plt.axis([0, 5000, 0, 0.01])
        # plt.legend()
        # plt.pause(0.5)

    count += 1  # Increment count
    # ^ unused?
        
    print(str(ind) + "/" + str(range(nBlock)))

    mask_KL = np.ones((nBlock, numChannels))  # Initialize mask_KL

    # Compute the mask for KL distance
    for chan in range(numChannels):
        for ind in range(nBlock):
            alpha = np.median(KL[:, chan])
            beta = np.median(np.abs(KL[:, chan] - alpha))
            if np.abs((KL[ind, chan] - alpha) / beta) > theta:
                mask_KL[ind, chan] = 0
            else:
                mask_KL[ind, chan] = 1

    # Masked KL matrix (optional)
    # masked_KL = KL * mask_KL

    # This needs moved to not be in the loop???
    # Visualize the KL matrix using a waterfall plot
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(nBlock), np.arange(numChannels))
    ax.plot_surface(X, Y, KL.T, cmap='viridis')
    plt.colorbar(ax.plot_surface(X, Y, KL.T, cmap='viridis'))
    plt.xlabel('Blocks grouped by 512 time samples')
    plt.ylabel('Channels, in bins')
    plt.title('Relative Entropy, pol0')
    plt.show()

    # Apply mask to PSD data
    burst = np.zeros((nBlock * nSub, numChannels))
    for ind in range(nBlock):
        for chan in range(numChannels):
            burst[ind * nSub:(ind + 1) * nSub, chan] = mask_KL[ind, chan] * psd_chunk[chan, ind * nSub:(ind + 1) * nSub]

    # Define burst parameters
    DM = 15.917
    BW = 800 * 10 ** 6  # bandwidth [Hz]
    f_c = 1500097656  # center frequency [Hz]
    K = 4096  # number of channels
    Ts = K / BW  # sample period [sec]
    flo = 1100
    fhi = 1900

    # 
    # Dedisperse the burst (assuming dedispersion is a custom function)
    def dedisperse(data, dm, f_lo, f_hi, tsamp):
        """
        Dedisperses a 2D array of data (frequency vs. time) given a dispersion measure (DM).

        Parameters:
            data: 2D NumPy array (frequency vs. time)
            dm: Dispersion measure (pc/cm^3)
            f_lo: Lowest frequency in the data (MHz)
            f_hi: Highest frequency in the data (MHz)
            tsamp: Time sampling interval (seconds)

        Returns:
            Dedispersed data
        """

        nchan, nsamp = data.shape
        freqs = np.linspace(f_lo, f_hi, nchan)

        dedispersed_data = np.zeros_like(data)
        for i in range(nchan):
            delay = 4.15e-3 * dm * (freqs[i] ** -2 - f_hi ** -2) / tsamp
            shift = int(np.round(delay))
            if shift > 0:
                dedispersed_data[i, shift:] = data[i, :-shift]
            elif shift < 0:
                dedispersed_data[i, :shift] = data[i, -shift:]
            else:
                dedispersed_data[i, :] = data[i, :]

        return dedispersed_data

    dedispersed_burst = dedisperse(burst.T, DM, flo, fhi, Ts)

    # Create a time array for high-resolution
    delta = 1
    time_high_res = np.arange(0, Ts * nSub - Ts * delta, Ts * delta, dtype=np.float32)

    # Calculate the SNR (intensity) of the single pulse
    intensity = np.sum(dedispersed_burst.astype(np.float32), axis=0)

    # Write intensity to a file
    with open("Intensity_J1713_Mat_0.txt", "w") as f:
        f.write(intensity)

    # Print shape of burst array
    print(burst.shape)
