from src.utils import Nii, NiiSeg, NSlice
from src.fit.fit import FitData
from pathlib import Path


class AppData:
    def __init__(self):
        self.app_path = Path(__file__).parent.parent
        self.nii_img: Nii = Nii()
        self.nii_seg: NiiSeg = NiiSeg()
        self.nii_img_masked: Nii = Nii()
        self.nii_dyn: Nii = Nii()
        self.fit_data = FitData()
        self.plt = dict()
        self.setup_plt_dict()

    def setup_plt_dict(self):
        self.plt["seg_color"] = list()
        self.plt["seg_alpha"] = float()
        self.plt["seg_line_width"] = float()
        self.plt["n_slice"] = NSlice(0)
