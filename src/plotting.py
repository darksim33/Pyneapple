import numpy as np
from PIL import Image, ImageOps, ImageFilter
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.axis as Axis
import matplotlib.patches as patches

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQT as FigureCanvas

from src.utils import Nii, NiiSeg
from src.fit.parameters import Parameters
from src.fit.fit import FitData


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


def show_pixel_signal(
    axis: Axis, canvas: FigureCanvas, data, fit_params: Parameters, pos: list
):
    y_data = data.nii_img.array[pos[0], pos[1], data.plt.nslice.value, :]
    x_data = np.squeeze(fit_params.b_values)
    axis.clear()
    axis.plot(x_data, y_data, ".")
    axis.set_xlabel("b-Values")
    canvas.draw()


def show_pixel_fit(axis: Axis, canvas: FigureCanvas, data, pos: list):
    number_slice = data.plt.nslice.value
    for idx, arg in enumerate(data.fit.fit_data.fit_results.d):
        if (pos[0], pos[1], number_slice) == arg[0]:
            d_values = arg[1]
            f_values = data.fit.fit_data.fit_results.f[idx]
            s_0 = data.fit.fit_data.fit_results.S0[idx]

    # get Y data
    d_values = data.fit.fit_data.fit_results.d
    # how to get information from array?
    x_data = np.squeeze(data.fit.fit_data.fit_params.b_values)
    axis.plot(x_data)
    canvas.draw()


def show_pixel_spectrum(axis: Axis, canvas: FigureCanvas, data, pos: list):
    y_data = data.nii_dyn.array[pos[0], pos[1], data.plt.nslice.value, :]
    n_bins = np.shape(y_data)
    x_data = np.geomspace(0.0001, 0.2, num=n_bins[0])
    axis.clear()
    axis.plot(x_data, y_data)
    axis.set_xscale("log")
    # axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("D (mm²/s)")
    canvas.draw()


def show_seg_spectrum(axis: Axis, canvas: FigureCanvas, data, number_seg: int):
    seg_idxs = data.Nii_mask.get_segIndizes(number_seg)
    y_data = np.zeros(data.Nii_dyn.shape(3))
    for idx in seg_idxs:
        y_data = y_data + data.Nii_dyn.array[idx(0), idx(1), data.plt.nslice.value, :]
    n_bins = np.shape(y_data)
    x_data = np.geomspace(0.0001, 0.2, num=n_bins[0])
    axis.clear()
    axis.plot(x_data, y_data)
    axis.set_xscale("log")
    # axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("D (mm²/s)")
    canvas.draw()


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
