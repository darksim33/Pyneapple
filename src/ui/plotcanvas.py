from PyQt6 import QtWidgets, QtCore
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from src.appdata import AppData


class CustomCanvas:
    def __init__(self):
        """
        Custom Canvas to manage figure and axis of a matplotlib object.
        """
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axis: Axes = self.figure.add_subplot(111)


class PlotLayout(QtWidgets.QVBoxLayout):
    spectrum: CustomCanvas
    decay: CustomCanvas

    def __init__(self, data: AppData):
        """
        Layout holding multiple CustomCanvas for signal plotting.
        """
        super().__init__()
        self.data = data
        self.color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        self.setup_ui()

    def setup_ui(self):
        # Setup decay Canvas
        """
        Setup Plot Layout.

        The setup_ui function is called by the __init__ function of the class.
        It creates a CustomCanvas object for each plot and adds it to the layout.
        The CustomCanvas objects are defined in custom_canvas.py.

        Parameters
        ----------
            self
                Refer to the object itself
        """
        # TODO: The X-Labels are still invisible
        self.decay = CustomCanvas()
        self.decay.canvas.setMinimumWidth(250)
        self.decay.canvas.setMinimumHeight(250)
        self.decay.axis.set_xlabel("b-Values")
        self.addWidget(self.decay.canvas)

        # Setup spectrum Canvas
        self.spectrum = CustomCanvas()
        self.decay.canvas.setMinimumWidth(250)
        self.decay.canvas.setMinimumHeight(250)
        self.spectrum.axis.set_xscale("log")
        self.spectrum.axis.set_xlabel("D (mm²/s)")
        self.addWidget(self.spectrum.canvas)

    def plot_pixel_decay(self, pos: list):
        """
        The plot_pixel_decay function plots the decay of a pixel in the image.

        Parameters
        ----------
            self
                Make the function a method of the class
            pos: list
                Specify the position of the pixel in the image
        """
        # Prepare Data
        y_data = self.data.nii_img.array[
            pos[0], pos[1], self.data.plt["n_slice"].value, :
        ]
        x_data = np.squeeze(self.data.fit_data.fit_params.b_values)
        self.decay.axis.clear()
        self.decay.axis.plot(x_data, y_data, ".", color=self.color)
        self.decay.axis.set_xlabel("b-Values")
        self.decay.canvas.draw()

    def plot_pixel_fit(self, pos: list):
        """
        The plot_pixel_fit function plots the fit of a single pixel.

        Parameters
        ----------
            self
                Access the attributes and methods of the class
            pos: list
                Get the position of the pixel that is clicked on
        Returns
        -------
            The fitted curve of a pixel

        """
        pixel_result = self.data.fit_data.fit_results.curve.get(
            (pos[0], pos[1], self.data.plt["n_slice"].value), None
        )
        if pixel_result is not None:
            # Prepare Data
            y_data = np.squeeze(pixel_result)
            x_data = np.squeeze(self.data.fit_data.fit_params.b_values)
            self.decay.axis.plot(x_data, y_data, color=self.color, alpha=1)
            self.decay.canvas.draw()

    def plot_pixel_spectrum(self, pos: list):
        """
        The plot_pixel_spectrum function plots the diffusion spectrum of a single pixel.

        Parameters
        ----------
            self
                Make the function a method of the class
            pos: list
                Specify the position of the pixel in x,y,z coordinates
        Returns
        -------
            A plot of the spectrum for a given pixel

        """
        # Prepare Data
        y_data = self.data.nii_dyn.array[
            pos[0], pos[1], self.data.plt["n_slice"].value, :
        ]
        n_bins = np.shape(y_data)
        x_data = np.geomspace(0.0001, 0.2, num=n_bins[0])
        self.spectrum.axis.clear()
        self.spectrum.axis.plot(x_data, y_data, color=self.color)
        self.spectrum.axis.set_xscale("log")
        self.spectrum.axis.set_xlabel("D (mm²/s)")
        self.spectrum.canvas.draw()
