from utils import *

files = [
    r"E:\home\Thomas\Sciebo\Projekte\Kidney\Kidney_Delta\data\pro3\delta_00\01_img\Delta_00_026_s3_mc.nii",
    r"E:\home\Thomas\Sciebo\Projekte\Kidney\Kidney_Delta\data\pro3\delta_00\01_img\Delta_00_046_s3_mc.nii",
    r"E:\home\Thomas\Sciebo\Projekte\Kidney\Kidney_Delta\data\pro3\delta_00\01_img\Delta_00_066_s3_mc.nii",
    r"E:\home\Thomas\Sciebo\Projekte\Kidney\Kidney_Delta\data\pro3\delta_00\01_img\Delta_00_086_s3_mc.nii",
    r"E:\home\Thomas\Sciebo\Projekte\Kidney\Kidney_Delta\data\pro3\delta_00\01_img\Delta_00_106_s3_mc.nii",
]
pmask = r"E:\home\Thomas\Sciebo\Projekte\Kidney\Kidney_Delta\data\mask_r\delta_00_sl3_mask_r.nii.gz"
mask = nifti_img(pmask)

for file in files:
    file = Path(file)
    img = nifti_img(file)
    nname = file.name.split(".")[0] + ".xlsx"
    img_masked = applyMask2Image(img, mask)
    data = Signal2CSV(img_masked, nname)
print("done")
