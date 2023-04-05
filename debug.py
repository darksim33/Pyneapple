# from PineappleUI import startAppUI
import nibabel as nib
from pathlib import Path

# p = Path(r"C:\Users\thitho01\Documents\Python\Projects\NNLSDynAPP\data\pat01_img.nii")
# startAppUI(p)

nii = Path(r"data\test9AV.nii")
niiImg = nib.load(nii)
array = niiImg.get_fdata()
affine = niiImg.affine
header = niiImg.header
header.set_data_dtype("i4")
arrayNew = array.astype(int)

newNii = nib.Nifti1Image(arrayNew, affine, header)
out = Path(r"data\test9AVnew.nii")
nib.save(newNii, out)
