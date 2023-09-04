from __future__ import annotations
from PyQt6 import QtWidgets, QtGui
from pathlib import Path


def create_context_menu(parent):
    parent.context_menu = QtWidgets.QMenu(parent)
    plt_menu = QtWidgets.QMenu("Plotting", parent)
    plt_menu.addAction(parent.plt_show)
    # not in use atm
    # plt_menu.addSeparator()
    # plt_menu.addAction(main_window.plt_DispType_SingleVoxel)
    # plt_menu.addAction(main_window.plt_DispType_SegSpectrum)

    parent.context_menu.addMenu(plt_menu)
    parent.context_menu.addSeparator()

    parent.save_slice = QtGui.QAction(
        text="Save slice...",
        parent=parent,
        icon=parent.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton
            )
    )
    parent.save_slice.triggered.connect(_save_slice)
    parent.context_menu.addAction(parent.save_slice)


def _save_slice(parent):
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
        parent.img_fig.savefig(file_path, bbox_inches="tight", pad_inches=0)
        print("Figure saved:", file_path)
