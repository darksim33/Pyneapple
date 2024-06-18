from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from PyQt6 import QtGui

if TYPE_CHECKING:
    from .pyneapple_ui import MainWindow


class Filter:
    @staticmethod
    def event_filter_image_canvas(parent: MainWindow, event):
        """
        Event Filter Handler.

        The event_filter function is used to filter events that are passed to the
        event_handler. This function is called by the event handler and should return
        True if it wants the event handler to process this event, or False if it wants
        the event handler to ignore this particular mouse click. The default behavior of
        this function is always returning True, which means all mouse clicks will be processed.

        Parameters
        ----------
            parent
                Refer to the current instance of a class
            event
                Get the mouse click position
        Returns
        -------
            A boolean value

        """
        if event.button == 1:
            # left mouse button
            if parent.data.nii_img.path:
                if event.xdata and event.ydata:
                    # check if point is on image
                    position = [round(event.xdata), round(event.ydata)]
                    # correct inverted y-axis
                    position[1] = parent.data.nii_img.array.shape[1] - position[1]
                    parent.image_axis.pos_label.setText(
                        f"({position[0]}, {position[1]})"
                    )
                    if parent.settings.value("plt_show", type=bool):
                        if (
                            isinstance(parent.data.fit_data.params.b_values, np.ndarray)
                            and parent.data.fit_data.params.b_values.size > 0
                        ):
                            parent.plot_layout.decay.x_data = (
                                parent.data.fit_data.params.b_values
                            )

                        if parent.data.plt["plt_type"] == "voxel":
                            # plot decay
                            parent.plot_layout.data = parent.data
                            parent.plot_layout.plot_pixel_decay(position)

                            if np.any(parent.data.nii_dyn.array):
                                parent.plot_layout.plot_pixel_fit(position)
                                parent.plot_layout.plot_pixel_spectrum(position)
                        elif parent.data.plt["plt_type"] == "segmentation":
                            parent.plot_layout.data = parent.data
                            parent.plot_layout.plot_pixel_decay(
                                position, "segmentation"
                            )
                            if np.any(parent.data.nii_dyn.array):
                                parent.plot_layout.plot_pixel_fit(position)
                                parent.plot_layout.plot_pixel_spectrum(position)

        elif event.button == 3:
            parent.context_menu_image.popup(QtGui.QCursor.pos())

    @staticmethod
    def event_filter_plot_decay(parent: MainWindow, event):
        if event.button == 3:
            parent.context_menu_plot_decay.popup(QtGui.QCursor.pos())
