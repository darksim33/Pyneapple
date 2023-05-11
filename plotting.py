import numpy as np
from utils import nii, nii_seg
from PIL import Image, ImageOps, ImageFilter
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class plotting(object):
    def overlayImage(
        img: nii,
        mask: nii_seg,
        slice: int = 0,
        alpha: int = 126,
        scaling: int = 2,
        color: str = "red",
    ) -> Image:
        if np.array_equal(img.size[0:3], mask.size[0:3]):
            _Img = img.rgba(slice).copy()
            if np.count_nonzero(mask.array[:, :, slice, :]) > 0:
                _Mask = mask.rgba(slice).copy()
                imgOverlay = ImageOps.colorize(
                    _Mask.convert("L"), black="black", white=color
                )
                alphamap = ImageOps.colorize(
                    _Mask.convert("L"), black="black", white=(alpha, alpha, alpha)
                )
                imgOverlay.putalpha(alphamap.convert("L"))
                _Img.paste(imgOverlay, [0, 0], mask=imgOverlay)
            _Img = _Img.resize([_Img.size[0] * scaling, _Img.size[1] * scaling])
            return _Img

    def show_PixelSpectrum(axis, Canvas, data):
        ydata = data.nii_dyn.array[
            data.plt.pos[0], data.plt.pos[1], data.plt.nslice.value, :
        ]
        nbins = np.shape(ydata)
        xdata = np.geomspace(0.0001, 0.2, num=nbins[0])
        axis.clear()
        axis.plot(xdata, ydata)
        axis.set_xscale("log")
        # axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("D (mm²/s)")
        Canvas.draw()

    def show_SegSpectrum(axis, Canvas, data, nSeg):
        seg_idxs = data.nii_mask.get_segIndizes(nSeg)
        ydata = np.zeros(data.nii_dyn.shape(3))
        for idx in seg_idxs:
            ydata = ydata + data.nii_dyn.array[idx(0), idx(1), data.plt.nslice.value, :]
        nbins = np.shape(ydata)
        xdata = np.geomspace(0.0001, 0.2, num=nbins[0])
        axis.clear()
        axis.plot(xdata, ydata)
        axis.set_xscale("log")
        # axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("D (mm²/s)")
        Canvas.draw()

    # def np2lbl(xpos: int, ypos: int, ysize: int, scaling: int):
    #     xpos_new = int(xpos / scaling)
    #     ypos_new = ysize - int(ypos / scaling)
    #     return [xpos_new, ypos_new]

    def lbl2np(xpos: int, ypos: int, ysize: int, scaling: int):
        xpos_new = int(xpos / scaling)
        # y Axis is inverted for label coordinates
        ypos_new = ysize - int(ypos / scaling) - 1
        return [xpos_new, ypos_new]


class plot:
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
        # if figure is not None:
        #     self._figure = figure
        # else:
        #     self._figure = plt.figure()

        # self._axis = axis if not None else self._figure.add_subplot(111)
        # if (axis is not None) and (figure is not None):
        #     self._axis = axis
        # else:
        #     self._axis = self._figure.add_subplot(111)

        self.xlims = [0.0001, 0.2]

    def draw(self):
        x = self._xdata
        y = self.ydata
        if self._axis is not None:
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
