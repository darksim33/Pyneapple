from multiprocessing import freeze_support
from pathlib import Path
from functools import partial
import numpy as np

from src.fit.ideal import IDEALParams, fit_ideal_new
from src.fit.parameters import IVIMParams
from src.fit.model import Model
from src.utils import Nii, NiiSeg


def list_files(directory: Path | str, pattern: str | None) -> list:
    if isinstance(directory, str):
        directory = Path(directory)
    file_list = []
    for file in directory.iterdir():
        if file.is_file() and pattern in file.__str__():
            file_list.append(file)
    return file_list


def test_ideal_ivim_multithreading():
    freeze_support()
    img = Nii(Path(r"../data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"../data/test_mask.nii.gz"))
    json = Path(
        Path(__file__).parent.parent,
        "./resources/fitting/default_params_ideal_test.json",
    )
    ideal_params = IDEALParams(json)
    result = fit_ideal_new(img, seg, ideal_params)
    print("Done")

    # assert True
