import numpy as np
from utils import Nii, Nii_seg
from PIL import Image, ImageOps, ImageFilter
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Plotting(object):
    # def overlay_image(
    #     img: Nii,
    #     mask: Nii_seg,
    #     slice: int = 0,
    #     alpha: int = 126,
    #     scaling: int = 2,
    #     color: str = "red",
    # ) -> Image:
    #     if np.array_equal(img.array.shape[0:3], mask.array.shape[0:3]):
    #         _Img = img.to_rgba_image(slice).copy()
    #         if np.count_nonzero(mask.array[:, :, slice, :]) > 0:
    #             _Mask = mask.to_rgba_image(slice).copy()
    #             imgOverlay = ImageOps.colorize(
    #                 _Mask.convert("L"), black="black", white=color
    #             )
    #             alphamap = ImageOps.colorize(
    #                 _Mask.convert("L"), black="black", white=(alpha, alpha, alpha)
    #             )
    #             imgOverlay.putalpha(alphamap.convert("L"))
    #             _Img.paste(imgOverlay, [0, 0], mask=imgOverlay)
    #         _Img = _Img.resize([_Img.size[0] * scaling, _Img.size[1] * scaling])
    #         return _Img

    def show_pixel_spectrum(axis, Canvas, data, pos):
        ydata = data.nii_dyn.array[pos[0], pos[1], data.plt.nslice.value, :]
        nbins = np.shape(ydata)
        xdata = np.geomspace(0.0001, 0.2, num=nbins[0])
        axis.clear()
        axis.plot(xdata, ydata)
        axis.set_xscale("log")
        # axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("D (mm²/s)")
        Canvas.draw()

    def show_seg_spectrum(axis, Canvas, data, nSeg):
        seg_idxs = data.Nii_mask.get_segIndizes(nSeg)
        ydata = np.zeros(data.Nii_dyn.shape(3))
        for idx in seg_idxs:
            ydata = ydata + data.Nii_dyn.array[idx(0), idx(1), data.plt.nslice.value, :]
        nbins = np.shape(ydata)
        xdata = np.geomspace(0.0001, 0.2, num=nbins[0])
        axis.clear()
        axis.plot(xdata, ydata)
        axis.set_xscale("log")
        # axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("D (mm²/s)")
        Canvas.draw()


class Plot:
    def __init__(
        self,
        figure: FigureCanvas | Figure = None,
        axis: plt.Axes | None = None,
        ydata: np.ndarray | None = None,
        xdata: np.ndarray | None = None,
    ):
        self.ydata = ydata
        self._xdata = xdata
        self._figure = figure
        self._axis = axis
        self.xlims = [0.0001, 0.2]

    def draw(self, clear_axis: bool = False):
        x = self._xdata.copy()
        y = self._ydata.copy()
        if self._axis is not None:
            if clear_axis:
                self._axis.clear()
            self._axis.set_xscale("log")
            self._axis.set_xlabel("D (mm²/s)")
            self._axis.plot(x, y)
            if type(self._figure) == FigureCanvas:
                self._figure.draw()
            # elif type(self._figure) == Figure:
            #     plt.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xscale("log")
            ax.set_xlabel("D (mm²/s)")
            ax.plot(x, y)
            plt.show()

    @property
    def axis(self):
        """Figure Axis handle"""
        return self._axis

    @property
    def ydata(self):
        """Y Axis Data"""
        return self._ydata

    @ydata.setter
    def ydata(self, ydata: np.ndarray | None = None):
        self._ydata = ydata
        if ydata is not None:
            nbins = ydata.shape[-1]
            self._xdata = np.geomspace(self.xlims[0], self.xlims[1], num=nbins)
        else:
            self._xdata = None

    @property
    def xdata(self):
        """X Axis Data"""
        return self._xdata

    @xdata.setter
    def xdata(self, xdata: np.ndarray):
        if self._ydata:
            if np.array_equal(self._ydata.shape, xdata.shape):
                self._xdata = xdata
            else:
                raise Exception(
                    "X and Y data don't have the same shape! X:{0}; Y:{1}".format(
                        self._ydata.shape, xdata.shape
                    )
                )
        else:
            raise Exception("No Y data set!")
