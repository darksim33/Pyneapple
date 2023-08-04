import numpy as np
from PIL import Image, ImageOps, ImageFilter
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from src.utils import Nii, Nii_seg

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
        y_data = data.nii_dyn.array[pos[0], pos[1], data.plt.nslice.value, :]
        n_bins = np.shape(y_data)
        x_data = np.geomspace(0.0001, 0.2, num=n_bins[0])
        axis.clear()
        axis.plot(x_data, y_data)
        axis.set_xscale("log")
        # axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("D (mm²/s)")
        Canvas.draw()

    def show_seg_spectrum(axis, Canvas, data, nSeg):
        seg_idxs = data.Nii_mask.get_segIndizes(nSeg)
        y_data = np.zeros(data.Nii_dyn.shape(3))
        for idx in seg_idxs:
            y_data = (
                y_data + data.Nii_dyn.array[idx(0), idx(1), data.plt.nslice.value, :]
            )
        n_bins = np.shape(y_data)
        x_data = np.geomspace(0.0001, 0.2, num=n_bins[0])
        axis.clear()
        axis.plot(x_data, y_data)
        axis.set_xscale("log")
        # axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("D (mm²/s)")
        Canvas.draw()


class Plot:
    def __init__(
        self,
        figure: FigureCanvas | Figure = None,
        axis: plt.Axes | None = None,
        y_data: np.ndarray | None = None,
        x_data: np.ndarray | None = None,
    ):
        self._y_data = y_data
        self._x_data = x_data
        self._figure = figure
        self._axis = axis
        self._x_lim = [0.0001, 0.2]

    def draw(self, clear_axis: bool = False):
        x = self._x_data.copy()
        y = self._y_data.copy()
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
    def y_data(self):
        """Y Axis Data"""
        return self._y_data

    @y_data.setter
    def y_data(self, y_data: np.ndarray | None = None):
        self._y_data = y_data
        if y_data is not None:
            n_bins = y_data.shape[-1]
            self._x_data = np.geomspace(self.x_lim[0], self.x_lim[1], num=n_bins)
        else:
            self._x_data = None

    @property
    def x_data(self):
        """X Axis Data"""
        return self._x_data

    @x_data.setter
    def x_data(self, x_data: np.ndarray):
        if self._y_data:
            if np.array_equal(self._y_data.shape, x_data.shape):
                self._x_data = x_data
            else:
                raise Exception(
                    "X and Y data don't have the same shape! X:{0}; Y:{1}".format(
                        self._y_data.shape, x_data.shape
                    )
                )
        else:
            raise Exception("No Y data set!")
