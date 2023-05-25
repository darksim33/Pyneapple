# from PineappleUI import startAppUI
from pathlib import Path
import utils as ut
import fitting, imantics
import matplotlib.pyplot as plt
import matplotlib.patches as patches  #
from PIL import Image

# from plotting import Plot
import numpy as np
from multiprocessing import freeze_support


# img = ut.Nii(Path(r"data/01_img.nii"))
img = ut.Nii(Path(r"data/pat16_img.nii.gz"))
seg = ut.Nii_seg(Path(r"data/pat16_seg_test.nii.gz"))
# seg = ut.Nii_seg(Path(r"data/01_prostate.nii.gz"))
dyn = ut.Nii(Path(r"data/01_img_AmplDyn.nii"))

seg_slice = seg.array[:, :, 16, 0]
# seg_slice = np.zeros((150, 150), dtype=np.uint8)
seg_slice[seg_slice != 3] = 0
mask = imantics.Mask(seg_slice)
polys = mask.polygons()
polygon = seg.get_polygon_patch_2D(1, 13)
fig = plt.figure()
ax = fig.gca()
ax.add_patch(polygon)
ax.set_ylim(0, 200)
ax.set_xlim(0, 200)
plt.show()
seg.show(13)

print("Done")

# class IndexTracker:
#     def __init__(self, ax, X):
#         self.index = 0
#         self.X = X
#         self.ax = ax
#         self.im = ax.imshow(self.X[:, :, self.index], cmap="gray")
#         self.update()

#     def on_scroll(self, event):
#         # print(event.button, event.step)
#         increment = 1 if event.button == "up" else -1
#         max_index = self.X.shape[-1] - 1
#         self.index = np.clip(self.index + increment, 0, max_index)
#         self.update()

#     def update(self):
#         self.im.set_data(self.X[:, :, self.index])
#         self.ax.set_title(f"Use scroll wheel to navigate\nindex {self.index}")
#         self.im.axes.figure.canvas.draw()


# fig, ax = plt.subplots()
# tracker = IndexTracker(ax, img.array[:, :, :, 1])
# fig.canvas.mpl_connect("scroll_event", tracker.on_scroll)
# plt.show()


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
# ADC = 1200 * 1e-5
# ADC = 0.1
# nPoints = 4000
# sampleSpacing = 10 / 4
# BValues = np.linspace(0.0, nPoints * sampleSpacing, nPoints, endpoint=False)

# Signal = 1 * np.exp(2 * np.pi * 1j * BValues * ADC)
# Spectrum = fft.fft(Signal)
# Spectrum = fft.fftshift(Spectrum)

# xFreq = fft.fftfreq(nPoints, sampleSpacing)
# xFreq = fft.fftshift(xFreq)
# # xFreq = dFit.fitParams.get_DValues()
# # imp = signal.unit_impulse(100, 81)
# # plt.plot(np.arange(-50, 50), imp)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xFreq, np.abs(Spectrum))
# ax.margins(0.1, 0.1)
# # ax.set_xlim(1 * 1e-4, 2 * 1e-1)
# ax.set_xscale("log")
# ax.grid(True)
# plt.show()


# print("test")
