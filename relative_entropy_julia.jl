using MAT
using Plots

# Load .mat file data
filename = "./Matlab Files/j1713_mat_0.mat"
data = matread(filename)

reData = data["re"]
imData = data["im"]

numChannels = 4096
timeSamples = 65024

sigma = 4
theta = sigma / 0.6745

scale = 0:(2^16 - 1)
v = 1:2^16

nSub = 512
nBlock = 127

No = 0
position = No * nBlock

re_chunk = Float32.(reData[:, position * nSub + 1:(position + nBlock) * nSub, 1])
im_chunk = Float32.(imData[:, position * nSub + 1:(position + nBlock) * nSub, 1])

psd_chunk = re_chunk .* re_chunk + im_chunk .* im_chunk

KL = zeros(Float64, nBlock, numChannels)

for ind in 1:nBlock
    # Extract the block of data
    tf = transpose(psd_chunk[:, (ind - 1) * nSub + 1:ind * nSub])

    for chan in 1:numChannels
        # Histogram of intensity per channel per segment
        num, pos = histogram(tf[:, chan], scale; normalize = true)
        num = num / sum(num) + 1e-10  # Normalize and avoid division by zero

        # Mean and variance per channel
        mu = mean(tf[:, chan])
        sig = var(tf[:, chan])

        # Reference PDF
        num_ref = exp.(-scale ./ sqrt(sig)) .- exp.(-v ./ sqrt(sig)) .+ 1e-10
        num_ref = num_ref[1:length(num)]

        # Relative entropy (KL divergence) per channel per segment
        KL[ind, chan] = sum(num_ref .* log.(num_ref ./ num)) + sum(num .* log.(num ./ num_ref))
    end

    println("$(ind)/$nBlock")
end

mask_KL = ones(Float64, nBlock, numChannels)

# Compute the mask for KL distance
for chan in 1:numChannels
    for ind in 1:nBlock
        alpha = median(KL[:, chan])
        beta = median(abs.(KL[:, chan] .- alpha))
        if abs((KL[ind, chan] - alpha) / beta) > theta
            mask_KL[ind, chan] = 0
        else
            mask_KL[ind, chan] = 1
        end
    end
end

# Visualize the KL matrix using a waterfall plot
X, Y = meshgrid(1:nBlock, 1:numChannels)
plot(surface(X, Y, KL'; cmap = :viridis), xlabel = "Blocks grouped by 512 time samples", ylabel = "Channels, in bins", title = "Relative Entropy, pol0", colorbar = true)

# Apply mask to PSD data
burst = zeros(Float64, nBlock * nSub, numChannels)
for ind in 1:nBlock
    for chan in 1:numChannels
        burst[(ind - 1) * nSub + 1:ind * nSub, chan] .= mask_KL[ind, chan] .* psd_chunk[chan, (ind - 1) * nSub + 1:ind * nSub]
    end
end

# Define burst parameters
DM = 15.917
BW = 800e6  # bandwidth [Hz]
f_c = 1500097656  # center frequency [Hz]
K = 4096  # number of channels
Ts = K / BW  # sample period [sec]
flo = 1100
fhi = 1900

# Dedisperse the burst (assuming dedispersion is a custom function)
function dedisperse(data, dm, f_lo, f_hi, tsamp)
    nchan, nsamp = size(data)
    freqs = range(f_lo, stop = f_hi, length = nchan)

    dedispersed_data = zeros(size(data))
    for i in 1:nchan
        delay = 4.15e-3 * dm * (freqs[i]^(-2) - f_hi^(-2)) / tsamp
        shift = round(Int, delay)
        if shift > 0
            dedispersed_data[i, shift + 1:end] = data[i, 1:end - shift]
        elseif shift < 0
            dedispersed_data[i, 1:end + shift] = data[i, -shift + 1:end]
        else
            dedispersed_data[i, :] = data[i, :]
        end
    end
    return dedispersed_data
end

dedispersed_burst = dedisperse(transpose(burst), DM, flo, fhi, Ts)

# Create a time array for high-resolution
delta = 1
time_high_res = Float32.(0:Ts * delta:(Ts * nSub - Ts * delta))

# Calculate the SNR (intensity) of the single pulse
intensity = sum(Float32.(dedispersed_burst), dims = 1)

# Write intensity to a file
open("Intensity_J1713_Mat_0.txt", "w") do f
    write(f, intensity)
end

# Print shape of burst array
println(size(burst))
