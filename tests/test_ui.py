import pytest
from PyQt6.QtCore import Qt

# from PyQt6 import QtWidgets
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QMessageBox

from pyneapple_ui.dlg_prompts import ReshapeSegMessageBox


@pytest.mark.ui
def test_reshape_seg_dlg(message_box, qtbot, img, seg):
    dialog = ReshapeSegMessageBox()
    button = dialog.button(QMessageBox.StandardButton.Yes)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    button = dialog.button(QMessageBox.StandardButton.No)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
