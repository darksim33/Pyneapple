import pytest
import time
import numpy as np

from scipy.optimize import curve_fit
from functools import partial
from pathlib import Path
from multiprocessing import freeze_support

from src.pyneapple.fit import IVIMParams
from src.pyneapple.fit import Model
from pyneapple.utils.nifti import Nii, NiiSeg
from src.pyneapple.fit import fit


@pytest.fixture
def nnls_fit_data():
    freeze_support()
    img = Nii(Path(r"data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"data/test_mask.nii.gz"))

    fit_data = fit.FitData(
        "NNLS", Path(r"resources/fitting/default_params_NNLS.json"), img, seg
    )
    fit_data.fit_params.max_iter = 10000

    return fit_data


@pytest.fixture
def mono_exp():
    freeze_support()
    img = Nii(Path(r"../data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"../data/test_mask.nii.gz"))
    fit_data = fit.FitData("MonoExp", img, seg)
    fit_data.fit_params = IVIMParams()
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
    model = IVIM_wrapper
    pixel_args = mono_exp.fit_params.get_element_args(
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
    fit.fit(fit_function, pixel_args, n_pools, False)
    assert True


def IVIM_wrapper(b_values, *args):
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
    max_iter,
):
    result = curve_fit(
        IVIM_wrapper,
        b_values.T,
        signal,
        args,
        bounds=(lb, ub),
        max_nfev=max_iter,
    )[0]
    return idx, result


def test_starmap_mono(mono_exp: fit.FitData):
    freeze_support()
    n_pools = 2

    x0 = np.array(
        [
            0.1,  # D_fast
            210,  # S_0
        ]
    )
    lb = np.array(
        [
            0.01,  # D_fast
            10,  # S_0
        ]
    )
    ub = np.array(
        [
            0.5,  # D_fast
            1000,  # S_0
        ]
    )

    b_values = np.array(
        [
            [
                0,
                5,
                10,
                20,
                30,
                40,
                50,
                75,
                100,
                150,
                200,
                250,
                300,
                400,
                525,
                750,
            ]
        ]
    )

    img = Nii(Path(r"data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"data/test_mask.nii.gz"))

    zip(
        ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))),
        (
            img.array[i, j, k, :]
            for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
        ),
    )
    # pixel_args = [_ for _ in pixel_args][:4]

    pixel_args = [
        _
        for _ in mono_exp.fit_params.get_element_args(
            mono_exp.img.array, mono_exp.seg.array
        )
    ][:4]
    fit_function = partial(
        mono,
        b_values=np.squeeze(b_values.T),
        args=x0,
        lb=lb,
        ub=ub,
        max_iter=200,
        TM=None,
    )
    fit.fit(fit_function, pixel_args, n_pools, False)
    assert True


def test_starmap_bi():
    freeze_support()
    n_pools = 2
    x0 = np.array(
        [
            0.1,  # D_fast
            0.005,  # D_inter
            0.1,  # f_fast
            210,  # S_0
        ]
    )
    lb = np.array(
        [
            0.01,  # D_fast
            0.003,  # D_inter
            0.01,  # f_fast
            10,  # S_0
        ]
    )
    ub = np.array(
        [
            0.5,  # D_fast
            0.01,  # D_inter
            0.7,  # f_fast
            1000,  # S_0
        ]
    )
    b_values = np.array(
        [
            [
                0,
                5,
                10,
                20,
                30,
                40,
                50,
                75,
                100,
                150,
                200,
                250,
                300,
                400,
                525,
                750,
            ]
        ]
    )

    img = Nii(Path(r"data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"data/test_mask.nii.gz"))

    pixel_args = zip(
        ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))),
        (
            img.array[i, j, k, :]
            for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
        ),
    )

    fit_function = partial(
        multi,
        b_values=np.squeeze(b_values.T),
        args=x0,
        lb=lb,
        ub=ub,
        max_iter=200,
        TM=None,
    )
    fit.fit(fit_function, pixel_args, n_pools, False)
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
        def mono_model(b_values: np.ndarray, s0, s1):
            f = np.array(s0 * np.exp(-np.kron(b_values, s1)))
            if TM:
                f *= np.exp(-args[2] / TM)
            return f

        return mono_model

    start_time = time.time()
    try:
        fit_result = curve_fit(
            mono_wrapper(TM),
            b_values,
            signal,
            p0=args,
            bounds=(lb, ub),
            max_nfev=max_iter,
        )[0]
    except (RuntimeError, ValueError):
        fit_result = np.zeros(args.shape)
    print(time.time() - start_time)
    return idx, fit_result


def bi(
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

    def bi_wrapper(TM: float | None):
        # def mono_model(b_values: np.ndarray, s0, s1, f1, s2, *args):
        def mono_model(b_values: np.ndarray, *args):
            # f = np.array(s0 * ((f1 * np.exp(-np.kron(b_values, s1)) + np.exp(-np.kron(b_values, s2)))))

            f = np.array(
                args[0]
                * (
                    (
                        args[2] * np.exp(-np.kron(b_values, args[1]))
                        + np.exp(-np.kron(b_values, args[3]))
                    )
                )
            )
            if TM:
                f *= np.exp(-args[2] / TM)
            return f

        return mono_model

    start_time = time.time()
    try:
        fit_result = curve_fit(
            bi_wrapper(TM),
            b_values,
            signal,
            p0=args,
            bounds=(lb, ub),
            max_nfev=max_iter,
        )[0]
    except (RuntimeError, ValueError):
        fit_result = np.zeros(args.shape)
    print(time.time() - start_time)
    return idx, fit_result


def multi(
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

    def bi_wrapper(TM: float | None, n_comps: int | None):
        # def mono_model(b_values: np.ndarray, s0, s1, f1, s2, *args):
        if n_comps == 2:

            def mono_model(b_values: np.ndarray, *args):
                # f = np.array(s0 * ((f1 * np.exp(-np.kron(b_values, s1)) + np.exp(-np.kron(b_values, s2)))))

                f = np.array(
                    args[0]
                    * (
                        (
                            args[2] * np.exp(-np.kron(b_values, args[1]))
                            + np.exp(-np.kron(b_values, args[3]))
                        )
                    )
                )
                if TM:
                    f *= np.exp(-args[2] / TM)
                return f

        elif n_comps == 1:

            def mono_model(b_values: np.ndarray, *args):
                f = np.array(args[0] * np.exp(-np.kron(b_values, args[1])))
                if TM:
                    f *= np.exp(-args[2] / TM)
                return f

        else:
            return None

        return mono_model

    start_time = time.time()
    try:
        fit_result = curve_fit(
            bi_wrapper(TM, n_comps=2),
            b_values,
            signal,
            p0=args,
            bounds=(lb, ub),
            max_nfev=max_iter,
        )[0]
        print(time.time() - start_time)
    except (RuntimeError, ValueError):
        fit_result = np.zeros(args.shape)
        print("Error")
    return idx, fit_result


def test_starmap_model_new():
    freeze_support()
    n_pools = 2
    x0 = np.array(
        [
            0.1,  # D_fast
            0.005,  # D_inter
            0.1,  # f_fast
            210,  # S_0
        ]
    )
    lb = np.array(
        [
            0.01,  # D_fast
            0.003,  # D_inter
            0.01,  # f_fast
            10,  # S_0
        ]
    )
    ub = np.array(
        [
            0.5,  # D_fast
            0.01,  # D_inter
            0.7,  # f_fast
            1000,  # S_0
        ]
    )
    b_values = np.array(
        [
            [
                0,
                5,
                10,
                20,
                30,
                40,
                50,
                75,
                100,
                150,
                200,
                250,
                300,
                400,
                525,
                750,
            ]
        ]
    )
    img = Nii(Path(r"data/test_img_176_176.nii"))
    seg = NiiSeg(Path(r"data/test_mask.nii.gz"))
    # fit_data = fit.FitData("MultiExp", img, seg)

    pixel_args = zip(
        ((i, j, k) for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))),
        (
            img.array[i, j, k, :]
            for i, j, k in zip(*np.nonzero(np.squeeze(seg.array, axis=3)))
        ),
    )
    # model = partial(Model.MultiExp.fit, n_components=2, mixing_time=None)
    Model.IVIM.wrapper(n_components=2, mixing_time=None)
    fit_function = partial(
        Model.IVIM.fit,
        b_values=np.squeeze(b_values.T),
        args=x0,
        lb=lb,
        ub=ub,
        max_iter=200,
        mixing_time=None,
        timer=True,
        n_components=2,
    )
    fit.fit(fit_function, pixel_args, n_pools, True)
    assert True
