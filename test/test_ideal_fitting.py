import pytest
from multiprocessing import freeze_support

from pyneapple.fit import FitData


@pytest.mark.skip("Not working properly atm.")
@pytest.mark.slow
def test_ideal_ivim_sequential(
    test_ideal_fit_data: FitData, out_nii, out_excel, capsys
):
    freeze_support()
    test_ideal_fit_data.fit_IDEAL(multi_threading=False)
    test_ideal_fit_data.results.save_results_to_excel(out_excel)
    test_ideal_fit_data.results.save_fitted_parameters_to_nii(
        out_nii, shape=test_ideal_fit_data.img.array.shape
    )
    capsys.readouterr()
    assert True


@pytest.mark.skip("Not working properly atm.")
@pytest.mark.slow
def test_ideal_ivim_multithreading(
    test_ideal_fit_data: FitData, out_nii, out_excel, capsys
):
    freeze_support()
    test_ideal_fit_data.fit_IDEAL(multi_threading=True)
    test_ideal_fit_data.results.save_results_to_excel(out_excel)
    test_ideal_fit_data.results.save_fitted_parameters_to_nii(
        out_nii, shape=test_ideal_fit_data.img.array.shape
    )
    capsys.readouterr()
    assert True
