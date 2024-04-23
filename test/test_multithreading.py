import pytest
import time
import numpy as np

from scipy.optimize import curve_fit
from functools import partial
from pathlib import Path
from multiprocessing import freeze_support

from pyneapple.fit import Model
from pyneapple.fit import FitData


def test_nnls_pixel_multi_reg_0(nnls_fit_data: FitData, capsys):
    nnls_fit_data.fit_params.reg_order = 0
    nnls_fit_data.fit_pixel_wise(multi_threading=True)
    capsys.readouterr()
    assert True


def test_nnls_pixel_multi_reg_2(nnls_fit_data: FitData, capsys):
    nnls_fit_data.fit_params.reg_order = 2
    nnls_fit_data.fit_params.n_pools = 2
    nnls_fit_data.fit_pixel_wise(multi_threading=True)
    capsys.readouterr()
    assert True


def test_nnls_pixel_multi_reg_3(nnls_fit_data: FitData, capsys):
    nnls_fit_data.fit_params.reg_order = 3
    nnls_fit_data.fit_params.n_pools = 2
    nnls_fit_data.fit_pixel_wise(multi_threading=True)
    capsys.readouterr()
    assert True


def test_nnls_pixel_multi_reg_cv(nnlscv_fit_data: FitData, capsys):
    nnlscv_fit_data.fit_pixel_wise(multi_threading=True)
    capsys.readouterr()
    assert True
