# dedisperse.jl
module dedispersion
    export dedisperse

    function dedisperse(data::Array{Float64, 2}, dm::Float64, f_lo::Float64, f_hi::Float64, tsamp::Float64)
        nchan, nsamp = size(data)
        freqs = range(f_lo, stop=f_hi, length=nchan)  # Equivalent to np.linspace
    
        dedispersed_data = zeros(size(data))
        for i in 1:nchan
            delay = 4.15e-3 * dm * (freqs[i]^(-2) - f_hi^(-2)) / tsamp
            shift = round(Int, delay)
    
            if shift > 0
                dedispersed_data[i, shift+1:end] .= data[i, 1:end-shift]
            elseif shift < 0
                dedispersed_data[i, 1:end+shift] .= data[i, 1-shift:end]
            else
                dedispersed_data[i, :] .= data[i, :]
            end
        end
    
        return dedispersed_data
    end
    
end