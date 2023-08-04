from pathlib import Path
from multiprocessing import freeze_support

from src.fit import fit
from src.fit.model import Model
from src.fit.ideal import ideal_fitting as IDEAL
from src.utils import Nii, Nii_seg


if __name__ == "__main__":
    freeze_support()
    img = Nii(Path(r"data/01_img.nii"))
    seg = Nii_seg(Path(r"data/01_prostate.nii.gz"))
    ideal_params = IDEAL.IDEALParams()
    ideal_params.model = Model.multi_exp()
    IDEAL.fit_ideal(img, ideal_params, seg)
