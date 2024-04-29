import sys
from PyQt6 import QtWidgets

from pyneapple.ui.dlg_prompts import ReshapeSegMessageBox


def test_reshape_seg_dlg(img, seg):
    app = QtWidgets.QApplication(sys.argv)
    dialog = ReshapeSegMessageBox(debug=True)
    dialog.exec()
    assert True
