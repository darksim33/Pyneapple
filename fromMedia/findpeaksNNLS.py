import math
import numpy as np
from scipy.signal import find_peaks, peak_widths


def findpeaksNNLS(s, DValues):
    # find peaks and diffusion coefficients of NNLS fitting results

    peaks = 3
    d_tot = f_tot = np.zeros((len(s), len(s), peaks))

    for i in range(len(s)):
        for j in range(len(s)):
            # TODO: descending output?!
            # TODO: thresholding possible?
            d, properties = find_peaks(s[i][j][:], height=0)
            fwhm = peak_widths(s[i][j][:], d, rel_height=0.5)
            maxima = properties["peak_heights"]

            # Convert back to log scale values
            d = DValues[d]

            # Calc area under gaussian curve
            f = (
                np.multiply(maxima, fwhm[0])
                / (2 * math.sqrt(2 * math.log(2)))
                * math.sqrt(2 * math.pi)
            )

            # Fill with zeros for suitable array size
            if len(d) < peaks:
                nz = peaks - len(d)
                d = np.append(d, np.zeros(nz))
                f = np.append(f, np.zeros(nz))

            # TODO: implement AUC calculation?

            # Threshold (remove entries with vol frac < 3%)
            d[f < 0.03] = 0
            # TODO: obsolet code if threshold/prominence adjusted in find_peaks
            f[f < 0.03] = 0

            f = np.divide(f, sum(f)) * 100  # normalize f

            d_tot[i][j][:] = d
            f_tot[i][j][:] = f

    return d_tot, f_tot
