# Dedisperse the burst (optimized with vectorization)
def dedisperse(data, dm, f_lo, f_hi, tsamp):
    nchan, nsamp = data.shape
    freqs = np.linspace(f_lo, f_hi, nchan)

    delay = 4.15e-3 * dm * (freqs ** -2 - f_hi ** -2) / tsamp
    shift = np.round(delay).astype(int)

    dedispersed_data = np.zeros_like(data)
    for i, sh in enumerate(shift):
        if sh > 0:
            dedispersed_data[i, sh:] = data[i, :-sh]
        elif sh < 0:
            dedispersed_data[i, :sh] = data[i, -sh:]
        else:
            dedispersed_data[i, :] = data[i, :]

    return dedispersed_data