import pytest
from pathlib import Path
from functools import wraps
from multiprocessing import freeze_support

from pyneapple.utils.nifti import Nii, NiiSeg
from pyneapple.fit import FitData


# Decorators
def freeze_me(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    freeze_support()
    return wrapper


# @pytest.mark.order(
#     after="test_nnls_parameters.py::TestNNLSParameters::test_nnls_init_parameters"
# )
class TestNNLSFitting:
    # Segmented sequential fitting
    @pytest.mark.parametrize("reg_order", [0, 1, 2, 3])
    def test_nnls_segmented_reg(
        self,
        capsys,
        reg_order,
        nnls_fit_data: FitData,
        out_nii: Path,
    ):
        nnls_fit_data.fit_params.reg_order = reg_order
        nnls_fit_data.fit_segmentation_wise()

        nii_dyn = Nii().from_array(
            nnls_fit_data.fit_results.spectrum.as_array(nnls_fit_data.seg.array.shape)
        )
        nii_dyn.save(out_nii, dtype=float)
        capsys.readouterr()
        assert True

    @pytest.mark.slow
    def test_nnls_segmented_reg_cv(
        self, capsys, nnlscv_fit_data: FitData, out_nii: Path
    ):
        nnlscv_fit_data.fit_segmentation_wise()

        nii_dyn = Nii().from_array(
            nnlscv_fit_data.fit_results.spectrum.as_array(
                nnlscv_fit_data.seg.array.shape
            )
        )
        nii_dyn.save(out_nii)
        capsys.readouterr()
        assert True

    # Multithreading
    @freeze_me
    @pytest.mark.slow
    @pytest.mark.parametrize("reg_order", [0, 1, 2, 3])
    def test_nnls_pixel_multi_reg(self, capsys, reg_order, nnls_fit_data: FitData):
        nnls_fit_data.fit_params.reg_order = reg_order
        nnls_fit_data.fit_pixel_wise(multi_threading=True)
        capsys.readouterr()
        assert True

    @freeze_me
    @pytest.mark.slow
    @pytest.mark.skip("Not working properly atm.")
    def test_nnls_pixel_multi_reg_cv(self, capsys, nnlscv_fit_data: FitData):
        nnlscv_fit_data.fit_pixel_wise(multi_threading=True)
        capsys.readouterr()
        assert True
