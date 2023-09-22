import pytest
from pathlib import Path
from multiprocessing import freeze_support
import numpy as np
from scipy.optimize import curve_fit
from functools import partial

from src.fit.parameters import MultiExpParams
from src.fit.model import Model
from src.utils import Nii, NiiSeg
from src.fit import fit


# TODO: checking for the correct calculation of the mean segmentation signal instead of fitting?!
@pytest.fixture
def nnls_fit_data():
    freeze_support()
    img = Nii(Path(r"../data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"../data/test_mask.nii.gz"))

    fit_data = fit.FitData("NNLS", img, seg)
    fit_data.fit_params.max_iter = 10000

    return fit_data


@pytest.fixture
def mono_exp():
    freeze_support()
    img = Nii(Path(r"../data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"../data/test_mask.nii.gz"))
    fit_data = fit.FitData("MonoExp", img, seg)
    fit_data.fit_params = MultiExpParams(n_components=1)
    fit_data.fit_params.boundaries.x0 = np.array(
        [
            0.1,  # D_fast
            210,  # S_0
        ]
    )
    fit_data.fit_params.boundaries.lb = np.array(
        [
            0.01,  # D_fast
            10,  # S_0
        ]
    )
    fit_data.fit_params.boundaries.ub = np.array(
        [
            0.5,  # D_fast
            1000,  # S_0
        ]
    )
    return fit_data


def test_nnls_pixel_multi_reg_0(nnls_fit_data: fit.FitData):
    nnls_fit_data.fit_params.reg_order = 0
    nnls_fit_data.fit_params.n_pools = 2
    nnls_fit_data.fit_pixel_wise(multi_threading=True)

    # nii_dyn = Nii().from_array(fit_data.fit_results.spectrum)
    # nii_dyn.save(r"nnls_pixel_multi_reg_0.nii")
    assert True


def test_tri_exp_pixel_multithreading(mono_exp: fit.FitData):
    mono_exp.fit_params.n_pools = 4
    mono_exp.model = Model.MultiTest()
    mono_exp.fit_pixel_wise(multi_threading=True)
    assert True


def test_tri_exp_basic(mono_exp):
    n_pools = 2
    model = multi_exp_wrapper
    pixel_args = mono_exp.fit_params.get_pixel_args(
        mono_exp.img.array, mono_exp.seg.array
    )
    fit_function = partial(
        fitter,
        b_values=mono_exp.fit_params.b_values,
        args=mono_exp.fit_params.boundaries.x0,
        lb=mono_exp.fit_params.boundaries.lb,
        ub=mono_exp.fit_params.boundaries.ub,
        model=model,
        max_iter=200,
    )
    results = fit.fit(fit_function, pixel_args, n_pools, False)
    assert True


def multi_exp_wrapper(b_values, *args):
    result = (
        np.exp(-np.kron(b_values, abs(args[0]))) * args[3]
        + np.exp(-np.kron(b_values, abs(args[1]))) * args[4]
        + np.exp(-np.kron(b_values, abs(args[2]))) * (1 - (np.sum(args[3:-1])))
    ) * args[-1]
    return result


def fitter(
    idx,
    signal,
    b_values,
    args: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    model,
    max_iter,
):
    result = curve_fit(
        multi_exp_wrapper,
        b_values.T,
        signal,
        args,
        bounds=(lb, ub),
        max_nfev=max_iter,
    )[0]
    return idx, result


def test_starmap_mono(mono_exp):
    n_pools = 2
    pixel_args = [
        _
        for _ in mono_exp.fit_params.get_pixel_args(
            mono_exp.img.array, mono_exp.seg.array
        )
    ][:4]
    fit_function = partial(
        mono,
        b_values=np.squeeze(mono_exp.fit_params.b_values.T),
        args=mono_exp.fit_params.boundaries.x0,
        lb=mono_exp.fit_params.boundaries.lb,
        ub=mono_exp.fit_params.boundaries.ub,
        max_iter=200,
        TM=None,
    )
    results = fit.fit(fit_function, pixel_args, n_pools, True)
    assert True


def mono(
    idx: int,
    signal: np.ndarray,
    b_values: np.ndarray,
    args: np.ndarray,
    TM: float | None,
    lb: np.ndarray,
    ub: np.ndarray,
    max_iter: int,
):
    """Mono exponential fitting model for ADC and T1"""
    # NOTE: does not theme to work for T1

    def mono_wrapper(TM: float | None):
        # TODO: use multi_exp(n_components=1) etc.
        def mono_model(b_values: np.ndarray, s0, s1):
            f = np.array(s0 * np.exp(-np.kron(b_values, s1)))
            if TM:
                f *= np.exp(-args[2] / TM)
            return f

        return mono_model

    fit = curve_fit(
        mono_wrapper(TM),
        b_values,
        signal,
        p0=args,
        bounds=(lb, ub),
        max_nfev=max_iter,
    )[0]
    return idx, fit


if __name__ == "__main__":
    test_starmap_mono(mono_exp)
