import sys
from pathlib import Path
from PyQt6 import QtWidgets

from src.ui.dialogues.prompt_dlg import ReshapeSegDlg
from src.utils import Nii, NiiSeg


def test_reshape_seg_dlg():
    QtWidgets.QApplication(sys.argv)
    img = Nii(Path(r"../data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"../data/test_mask_128.nii.gz"))

    dialog = ReshapeSegDlg(img, seg)
    dialog.exec()
    assert True
