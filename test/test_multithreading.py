import pytest
import time
import numpy as np

from scipy.optimize import curve_fit
from functools import partial
from pathlib import Path
from multiprocessing import freeze_support

from pyneapple.fit.parameters import IVIMParams
from pyneapple.fit import Model
from pyneapple.fit import FitData

freeze_support()


def test_ivim_mono_pixel_multithreading(ivim_mono_fit_data: FitData):
    ivim_mono_fit_data.fit_params.n_pools = 4
    ivim_mono_fit_data.fit_pixel_wise(multi_threading=True)
    assert True


def test_ivim_bi_pixel_multithreading(ivim_bi_fit_data: FitData):
    ivim_bi_fit_data.fit_params.n_pools = 4
    ivim_bi_fit_data.fit_pixel_wise(multi_threading=True)
    assert True


def test_ivim_tri_pixel_multithreading(ivim_tri_fit_data: FitData):
    ivim_tri_fit_data.fit_params.n_pools = 4
    ivim_tri_fit_data.fit_pixel_wise(multi_threading=True)
    assert True


def test_nnls_pixel_multi_reg_0(nnls_fit_data: FitData):
    nnls_fit_data.fit_params.reg_order = 0
    nnls_fit_data.fit_pixel_wise(multi_threading=True)
    assert True


def test_nnls_pixel_multi_reg_2(nnls_fit_data: FitData):
    nnls_fit_data.fit_params.reg_order = 2
    nnls_fit_data.fit_params.n_pools = 2
    nnls_fit_data.fit_pixel_wise(multi_threading=True)
    assert True


def test_nnls_pixel_multi_reg_3(nnls_fit_data: FitData):
    nnls_fit_data.fit_params.reg_order = 3
    nnls_fit_data.fit_params.n_pools = 2
    nnls_fit_data.fit_pixel_wise(multi_threading=True)
    assert True


def test_nnls_pixel_multi_reg_cv(nnlscv_fit_data: FitData):
    nnlscv_fit_data.fit_pixel_wise(multi_threading=True)
    assert True
