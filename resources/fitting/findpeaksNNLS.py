import math
import numpy as np
from scipy.signal import find_peaks, peak_widths


def findpeaksNNLS(signal, bins):
    # find peaks and diffusion coefficients of NNLS fitting results

    peaks = 3  # TODO: adjust
    d = f = np.zeros((len(signal), len(signal), peaks))

    for i in range(len(signal)):
        for j in range(len(signal)):
            # TODO: descending output?!
            # TODO: thresholding possible?
            idx, properties = find_peaks(signal[i][j][:], height=0)
            fwhm = peak_widths(signal[i][j][:], idx, rel_height=0.5)
            maxima = properties["peak_heights"]

            # Convert back to log scale values
            d_i = bins[idx]

            # Calc area under gaussian curve
            f_i = (
                np.multiply(maxima, fwhm[0])
                / (2 * math.sqrt(2 * math.log(2)))
                * math.sqrt(2 * math.pi)
            )

            # Fill with zeros for suitable array size
            if len(d_i) < peaks:
                nz = peaks - len(d_i)
                d_i = np.append(d_i, np.zeros(nz))
                f_i = np.append(f_i, np.zeros(nz))

            # TODO: implement AUC calculation?

            # Threshold (remove entries with vol frac < 3%)
            # TODO: obsolet code if threshold/prominence adjusted in find_peaks
            d_i[f_i < 0.03] = 0
            f_i[f_i < 0.03] = 0

            f_i = np.divide(f_i, sum(f_i)) * 100  # normalize f_i

            d[i][j][:] = d_i
            f[i][j][:] = f_i

    return d, f
