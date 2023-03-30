import sys
import utils
from PyQt6 import QtWidgets

# initiate QtWidget
app = QtWidgets.QApplication(sys.argv)

img_name = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image to mask:")[0]
mask_name = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image to mask:")[0]
1
img_nii = utils.nifti_img(img_name)
mask_nii = utils.nifti_img(mask_name)

img_masked = utils.overlayImage(img_nii, mask_nii).show()
