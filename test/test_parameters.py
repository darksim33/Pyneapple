from pathlib import Path
from functools import partial
from pyneapple.fit.parameters import Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams, JsonImporter


def test_init_parameters():
    Parameters()
    assert True


def test_init_nnls_parameters():
    NNLSParams()
    assert True


def test_init_nnls_cv_parameters():
    NNLSCVParams()
    assert True


def test_init_ivim_parameters():
    IVIMParams()
    assert True


def test_init_ideal_parameters():
    IDEALParams()
    assert True


def test_json_importer():
    params = JsonImporter(Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json"))
    isinstance(params, IVIMParams)
    assert True


def test_json_load():
    IVIMParams().load_from_json(Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json"))
    NNLSParams().load_from_json(Path(r"../src/pyneapple/resources/fitting/default_params_NNLS.json"))
    NNLSCVParams().load_from_json(Path(r"../src/pyneapple/resources/fitting/default_params_NNLSCV.json"))
    IDEALParams().load_from_json(Path(r"../src/pyneapple/resources/fitting/default_params_IDEAL_bi.json"))
    assert True


def test_json_save(capsys):
    # TODO: Save load compare
    # Test IVIM
    ivim_params = IVIMParams(Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json"))
    ivim_params.save_to_json(Path(r"test_params.json"))
    test_params = IVIMParams(Path(r"test_params.json"))

    attributes = compare_parameters(ivim_params, test_params)

    for attr in attributes:
        if not getattr(ivim_params, attr) == getattr(test_params, attr):
            ValueError(f"{attr} is not a valid parameter")

    params = IVIMParams(Path(r"../src/pyneapple/resources/fitting/default_params_IVIM_tri.json"))
    params.save_to_json(Path(r"test_params.json"))

    params = NNLSParams(Path(r"../src/pyneapple/resources/fitting/default_params_NNLS.json"))
    params.save_to_json(Path(r"test_params_nnls.json"))
    params = NNLSCVParams(Path(r"../src/pyneapple/resources/fitting/default_params_NNLSCV.json"))
    params.save_to_json(Path(r"test_params_nnls_reg.json"))
    capsys.readouterr()
    assert True


def compare_parameters(params1: IVIMParams, params2: IVIMParams) -> list:
    # compare attributes first
    attributes = [
        attr
        for attr in dir(params1)
        if not callable(getattr(params1, attr))
           and not attr.startswith("_")
           and not isinstance(getattr(params1, attr), partial)
    ]
    test_attributes = [attr
                       for attr in dir(params2)
                       if not callable(getattr(params2, attr))
                       and not attr.startswith("_")
                       and not isinstance(getattr(params2, attr), partial)]

    if not attributes == test_attributes:
        raise ValueError(f"Parameters attributes do not match!")

    return attributes

    #
    # for attr in attributes:
    #     if not getattr(params1, attr) == getattr(params2, attr):
    #         ValueError(f"{attr} is not a valid parameter")
