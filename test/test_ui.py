import sys
from pathlib import Path
from PyQt6 import QtWidgets

from src.ui.promptdlgs import ReshapeSegDlg
from src.utils import Nii, NiiSeg


def test_reshape_seg_dlg():
    app = QtWidgets.QApplication(sys.argv)
    img = Nii(Path(r"../data/test_img.nii"))
    seg = NiiSeg(Path(r"../data/test_mask_128.nii.gz"))

    dialog = ReshapeSegDlg(img, seg)
    result = dialog.exec()
    assert True
