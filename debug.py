# from PineappleUI import startAppUI
from pathlib import Path
import utils as ut
import fitting
import matplotlib.pyplot as plt

# from plotting import *
import numpy as np
from multiprocessing import freeze_support


img = ut.nii(Path(r"data/01_img.nii"))
seg = ut.nii_seg(Path(r"data/01_prostate.nii.gz"))
dyn = ut.nii(Path(r"data/01_img_AmplDyn.nii"))

# myplt = plot()
# idx = seg.get_segIndizes(1)[:, 0]
# myplt.ydata = dyn.array[idx[0], idx[1], idx[2], :]

print("Done")


# if __name__ == "__main__":
#     freeze_support()

#     fit_data = fitting.fitData("mono")
#     nii = Path(r"data/kid_img.nii")
#     fit_data.img.load(nii)
#     nii = Path(r"data/kid_mask.nii.gz")
#     fit_data.mask.load(nii)
#     # signal = np.array(
#     #     [
#     #         19.0,
#     #         19.0,
#     #         20.0,
#     #         27.0,
#     #         23.0,
#     #         23.0,
#     #         25.0,
#     #         15.0,
#     #         20.0,
#     #         15.0,
#     #         13.0,
#     #         14.0,
#     #         11.0,
#     #         14.0,
#     #         13.0,
#     #         19.0,
#     #     ]
#     # )
#     # bvalues = np.array(
#     #     [0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 525, 750]
#     # )
#     # test = fitting.fitModels.monoFit(
#     #     (1, 1, 1), signal, bvalues, [50, 0.001], [10, 0.0001], [1000, 0.01]
#     # )

#     fit_data.fitParams = fitting.MonoParams("mono")
#     fit_data.fitParams.bValues = np.array(
#         [
#             0,
#             5,
#             10,
#             20,
#             30,
#             40,
#             50,
#             75,
#             100,
#             150,
#             200,
#             250,
#             300,
#             400,
#             525,
#             750,
#         ]
#     )
#     fit_data.fitParams.variables.TM = 9.8
#     result = fitting.setupFitting(fit_data, debug=True)


from scipy import signal
from scipy import fft
import matplotlib.pyplot as plt
import numpy as np

# dFit = fitting.fitData(None, None, None)
# dFit.fitResults.Ds = ([0, 0, 0], np.array([0.1, 0.125]))
# dFit.fitResults.Fs = ([0, 0, 0], np.array([0.25, 0.75]))
# dFit.set_SpectrumFromVariables()
# Spectrum = dFit.spectrum
ADC = 1200 * 1e-5
ADC = 0.1
nPoints = 4000
sampleSpacing = 10 / 4
BValues = np.linspace(0.0, nPoints * sampleSpacing, nPoints, endpoint=False)

Signal = 1 * np.exp(2 * np.pi * 1j * BValues * ADC)
Spectrum = fft.fft(Signal)
Spectrum = fft.fftshift(Spectrum)

xFreq = fft.fftfreq(nPoints, sampleSpacing)
xFreq = fft.fftshift(xFreq)
# xFreq = dFit.fitParams.get_DValues()
# imp = signal.unit_impulse(100, 81)
# plt.plot(np.arange(-50, 50), imp)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xFreq, np.abs(Spectrum))
ax.margins(0.1, 0.1)
# ax.set_xlim(1 * 1e-4, 2 * 1e-1)
ax.set_xscale("log")
ax.grid(True)
plt.show()


print("test")
