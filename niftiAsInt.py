import sys  # , os
import nibabel as nib
from PyQt6 import QtWidgets

# from utils import *


def nii2int32(nii: nib.nifti1):
    nii_new = nib.Nifti1Image(nii.get_fdata().astype("int32"), nii.affine, nii.header)
    nii_new.set_data_dtype(8)  # https://brainder.org/2012/09/23/the-nifti-file-format/
    return nii_new


app = QtWidgets.QApplication(sys.argv)
fname = QtWidgets.QFileDialog.getOpenFileName()
nii = nib.load(fname[0])
nii_new = nii2int32(nii)
nib.save(nii_new, "new_nifti.nii")
print("done")
