from PyQt6 import QtWidgets, QtGui
from pathlib import Path


def create_context_menu(parent):
    """
    A context menu for the image viewer

    The create_context_menu function creates a context menu for the main window.

    Parameters
    ----------
        parent>
            Pass the main_window object to this function

    Returns
    -------
        A context menu for the image viewer
    """
    parent.context_menu = QtWidgets.QMenu(parent)
    plt_menu = QtWidgets.QMenu("Plotting", parent=parent.context_menu)
    plt_menu.addAction(parent.view_menu.plt_show)
    # not in use atm
    # plt_menu.addSeparator()
    # plt_menu.addAction(main_window.plt_DispType_SingleVoxel)
    # plt_menu.addAction(main_window.plt_DispType_SegSpectrum)

    parent.context_menu.addMenu(plt_menu)
    parent.context_menu.addSeparator()

    parent.save_slice = QtGui.QAction(
        text="Save slice...",
        parent=parent,
        icon=QtGui.QIcon(
            Path(
                Path(parent.data.app_path), "resources", "images", "Camera.ico"
            ).__str__()
        ),
        # icon=parent.style().standardIcon(
        #     QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
        # ),
    )
    parent.save_slice.triggered.connect(lambda x: _save_slice(parent))
    parent.context_menu.addAction(parent.save_slice)


def _save_slice(parent):
    """
    Save Slice to PNG Callback.

    The _save_slice function is called when the user clicks on the &quot;Save slice&quot; button.
    It opens a file dialog to allow the user to select where they want to save their image, and then saves it as a PNG
    file.


    Parameters
    ----------
        parent
            Access the parent object, which is the main window

    Returns
    -------
        The file path to the saved image
    """
    if parent.data.nii_img.path:
        file_name = parent.data.nii_img.path
        new_name = file_name.parent / (file_name.stem + ".png")

        file_path = Path(
            QtWidgets.QFileDialog.getSaveFileName(
                parent,
                "Save slice image:",
                new_name.__str__(),
                "PNG Files (*.png)",
            )[0]
        )
    else:
        file_path = None

    if file_path:
        parent.image_axis.figure.savefig(file_path, bbox_inches="tight", pad_inches=0)
        print("Figure saved:", file_path)
