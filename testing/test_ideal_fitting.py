from pathlib import Path

from multiprocessing import freeze_support

dir()
# from src.fit import fit
# from ..fitModel import Model
# from ..ideal import ideal_fitting as IDEAL
# from ..utils import *


# if __name__ == "__main__":
#     freeze_support()
#     img = ut.Nii(Path(r"data/01_img.nii"))
#     seg = ut.Nii_seg(Path(r"data/01_prostate.nii.gz"))    
#     ideal_params = IDEAL.IDEALParams()
#     ideal_params.model = Model.multi_exp()
#     IDEAL.fit_ideal(img, ideal_params, seg)