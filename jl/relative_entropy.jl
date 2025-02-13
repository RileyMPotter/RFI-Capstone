# relative_entropy.jl
include("jl/dedisperse.jl");

using .dedispersion
using MAT
using Plots

# Read data from .mat file
data = matread("./Matlab Files/j1713_mat_0.mat")

numChannels = 4096
timeSamples = 65024
sigma = 4
theta = sigma / 0.6745
scale = np.arange(0, 2 ^ 16)
v = np.arange(1, 2 ^ 16 + 1)
nSub = 512
nBlock = 127
No = 0
position = No * nBlock

# Define Burst Parameters
# Define burst parameters
DM = 15.917
BW = 800 * 10 ^ 6  # bandwidth [Hz]
f_c = 1500097656  # center frequency [Hz]
K = 4096  # number of channels
Ts = K / BW  # sample period [sec]
flo = 1100
fhi = 1900

count = 1;

reData = file["re"]
imdata = data["im"]

# Extract a chunk of the data
reChunk = Float32.(reData[:, position * nSub + 1 : (position + nBlock) * nSub, 1])
imChunk = Float32.(imData[:, position * nSub + 1 : (position + nBlock) * nSub, 1])

# Compute the PSD Chinuk
psdChunk = reChunk .* reChunk .+ imChunk .* imChunk

# Clear file, reData/Chunk and imData/Chunk
file = nothing
reData = nothing
reChunk = nothing
imData = nothing
imChunk = nothing

# Force Garbage Colleection
GC.gc()

# Parameters used to fit a curve in the histogram of intesnsities per channel
# scale = 0:1:2^16-1
# v = 1:2^16

KL = zeros(nBlock, numChannels) # Loop over blocks

for ind in 1:nBlock
    tf = psdCunk[:, (ind - 1) * nSub + 1 : ind * nSub]'

    for chan in 1:numChannels #Loop over channels
        # Histogram of intensity per channel per segment
        num, pos = StatsBase.Histogram(tf[:, chan], bins=scale, normalized=true)
        num = num ./ sum(num) .+ 10.0^(-10)  # Normalize and avoid division by zero

        # Mean and Variance per channel
        mu = mean(tf[:, chan])
        sig = var(tf[:, chan])

        # Reference PDF
        num_ref = exp.(-scale ./ sqrt(sig)) .- exp.(-v ./ sqrt(sig)) .+ 10.0^(-10)
        num_ref = num_ref[1:length(num)]

        # Relative entropy (KL divergence) per channel per segment
        KL[ind, chan] = sum(num_ref .* log.(num_ref ./ num)) + sum(num .* log.(num ./ num_ref))
    end
end

# Initialize maskKL
maskKl = ones(Floate64, nBlock, numChannels)

# Compute the mask for KL Distance
for chan in 1:numChannels
    for ind in 1:nBlock
        alpha = median(KL[:, chan])
        beta = median(abs.(KL[:, chan] .- alpha))
        if abs((KL[ind, chan] - alpha) / beta) > theta
            maskKL[ind, chan] = 0
        else
            maskKL[ind, chan] = 1
        end
    end
end

# Apply Mask to PSD data
burst = zeros(Float64, nBlock * nSub, numChannels)
for ind in 1:nBlock
    for chan in 1:numChannels
        burst[(ind - 1) * nSub + 1 : ind * nSub, chan] = mask_KL[ind, chan] .* psd_chunk[chan, (ind - 1) * nSub + 1 : ind * nSub]  
    end
end

dedispersedBurst = dedipserse(burst', DM, flo, fhi, Ts)

# Create a time array for high resolution
delta = 1
timeHighRes = Float32.(0:Ts * delta:Ts * nSub - Ts * delta)

# Calculatee the SNR (intensity) of the signle pulse
intensity = sum(FLoat32.(dedispersedBurst), dims = 1) # Sum along rows (dimension 1)

# Write all elements of the intensity array to a file
open("Intesity_J1713_Mat_0.txt", "w") do f
    write(f, string(intensity))
end

# Print the shape of the burst array
println("Burst array shape: ", size(burst))

# Print completed blocks
println("nBlocks Completed: $count")
count += 1  # Increment count

# Set the bacckend to PlotlyJS for 3D Plotting
plotlyjs()

# Generate data
X, Y = ndgrid(0:nBlock-1, 0:numChannels-1) #Equivalent to np.meshgrid
Z = KL' # Transpose KL

# Create a 3D Surface PlotlyJS
p = surface(
    X, Y, Z,
    color=:viridis, # Colormap
    xlabel="Blocks grouped by 512 time samples",
    ylabel="Channels, in bins",
    zlabel="Relative Entropy",
    title="Relative Entropy, pol0"
)

# Save the figure to a file
savefig(p, relative_entropy.png)

# Show the plot (opens in a browser with PlotlyJS Backend)
display(p)