import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io

# Start timing
tic = time.time()

# Parameters
DM = 15.917
BW = 800 * 10**6    # Bandwidth [Hz]
f_c = 1500097656    # Center frequency [Hz]
K = 4096            # Number of channels
Ts = K / BW         # Sample period [sec]
flo = 1100
fhi = 1900

# Load PSD data
mat_data = scipy.io.loadmat('mat_0_chunk_0_SW_Th_00001.mat')
burst = mat_data['burst']
K, Nsub = burst.shape

# Dedispersion function definition
def dedispersion(burst, DM, BW, f_c, K, Ts):
    # Placeholder implementation for dedispersion
    # This should be replaced with a real dedisperion algorithm
    return burst

# Dedisperse the burst
delta = 1
dedispersed_burst = dedispersion(burst, DM, BW, f_c, K, Ts)

# Plot dedispersed data
plt.figure(4)
time_axis = np.arange(0, Ts * delta * K, Ts * delta)
freq_axis = np.linspace(fhi, flo, K)
plt.imshow(dedispersed_burst, aspect='auto', extent=[time_axis[0], time_axis[-1], freq_axis[-1], freq_axis[0]], cmap='jet')
plt.title('Dedispered Data')
plt.gca().invert_yaxis()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Amplitude')
plt.show()

# Optional integration in time
tf_int = dedispersed_burst  # No integration since delta is 1

# Save high-resolution time data
time_high_res = np.arage(0, Ts * delta * Nsub, Ts * delta, dtype=np.float32)
np.savetxt()