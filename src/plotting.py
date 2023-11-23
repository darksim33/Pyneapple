import numpy as np
from pathlib import Path
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.axis as Axis

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import FigureCanvasQT as FigureCanvas

from src.fit.parameters import Parameters
from src.fit.fit import FitData
from src.appdata import AppData


def show_pixel_signal(
    axis: Axis, canvas: FigureCanvas, data: AppData, fit_params: Parameters, pos: list
):
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    y_data = data.nii_img.array[pos[0], pos[1], data.plt["n_slice"].value, :]
    x_data = np.squeeze(fit_params.b_values)
    axis.clear()
    axis.plot(x_data, y_data, ".", color=color)
    axis.set_xlabel("b-Values")
    canvas.draw()


def show_pixel_fit(axis: Axis, canvas: FigureCanvas, data: AppData, pos: list):
    number_slice = data.plt["n_slice"].value
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    # pixel_result = data.fit_data.fit_results.raw.get((pos[0], pos[1], number_slice), None)
    pixel_result = data.fit_data.fit_results.curve.get(
        (pos[0], pos[1], number_slice), None
    )
    if pixel_result is not None:
        # get Y data
        y_data = np.squeeze(pixel_result)
        # y_data = np.squeeze(data.fit_data.fit_params.fit_model(data.fit_data.fit_params.b_values, *pixel_result).T)
        # how to get information from array?
        x_data = np.squeeze(data.fit_data.fit_params.b_values)
        axis.plot(x_data, y_data, color=color, alpha=1)
        canvas.draw()


def show_pixel_spectrum(axis: Axis, canvas: FigureCanvas, data: AppData, pos: list):
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    y_data = data.nii_dyn.array[pos[0], pos[1], data.plt["n_slice"].value, :]
    n_bins = np.shape(y_data)
    x_data = np.geomspace(0.0001, 0.2, num=n_bins[0])
    axis.clear()
    axis.plot(x_data, y_data, color=color)
    axis.set_xscale("log")
    # axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("D (mm²/s)")
    canvas.draw()


def show_seg_spectrum(axis: Axis, canvas: FigureCanvas, data, number_seg: int):
    seg_idx = data.Nii_mask.get_segIndizes(number_seg)
    y_data = np.zeros(data.Nii_dyn.shape(3))
    for idx in seg_idx:
        y_data = (
            y_data + data.Nii_dyn.array[idx(0), idx(1), data.plt["n_slice"].value, :]
        )
    n_bins = np.shape(y_data)
    x_data = np.geomspace(0.0001, 0.2, num=n_bins[0])
    axis.clear()
    axis.plot(x_data, y_data)
    axis.set_xscale("log")
    # axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("D (mm²/s)")
    canvas.draw()


def create_heatmaps(data: FitData, d: dict, f: dict):
    n_comps = 3  # Take information out of model dict?!
    slice_number = 1  # middle slice
    img_dim = data.img.array.shape[0:3]
    model = data.model_name

    # Create 4D array heatmaps containing d and f values
    d_heatmap = np.zeros(np.append(img_dim, n_comps))
    f_heatmap = np.zeros(np.append(img_dim, n_comps))

    for key, value in d.items():
        d_heatmap[key, :] = value
        f_heatmap[key, :] = f[key]

    # Plot heatmaps
    fig, axs = plt.subplots(2, n_comps)
    fig.suptitle(f"{model}", fontsize=20)

    for comp in range(0, n_comps):
        axs[0, comp].imshow(d_heatmap[:, :, slice_number, comp])
        axs[1, comp].imshow(f_heatmap[:, :, slice_number, comp])

    # plt.show()
    fig.savefig(Path(f"data/results/heatmaps_{model}_slice_{slice_number}.png"))


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
